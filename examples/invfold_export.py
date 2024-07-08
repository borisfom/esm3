import os
import torch
import time
import torch.nn.functional as F

from esm.pretrained import (
    ESM3_sm_open_v0,
    ESM3_structure_encoder_v0,
)

from esm.models.esm3 import ESM3

from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
)

from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.structure.normalize_coordinates import (
    normalize_coordinates,
)
from esm.utils.structure.protein_chain import ProteinChain

from polygraphy.json import to_json, from_json, save_json
from torch.onnx import verification
from esm.utils.trt_utils import TRTWrapper
from esm.utils.stack_wrapper import TransformerStackWrapper

def get_structure_inputs_for_chain(
    chain: ProteinChain,
    should_normalize_coordinates: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor(chain.atom37_positions, dtype=torch.float32)
    plddt = torch.ones((coords.shape[0],), dtype=torch.float32)
    coords = torch.nn.functional.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = torch.nn.functional.pad(plddt, (1, 1), value=0)
    if should_normalize_coordinates:
        coords = normalize_coordinates(coords)
    return coords.unsqueeze(0), plddt.unsqueeze(0)


class Invfold(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ESM3_sm_open_v0("cpu")
        self.encoder = ESM3_structure_encoder_v0("cpu")
        self.tokenizer = EsmSequenceTokenizer()

    def encode(self, coords):
        # t0 = time.time()
        _, structure_tokens = self.encoder.encode(coords)
        # t1=time.time()
        # print(f"Encoder time: {t1-t0}")
        structure_tokens[:, 0] = 4098
        structure_tokens[:, -1] = 4097
        return structure_tokens
        
    def forward(
            self,
            coords: torch.Tensor,
            plddt: torch.Tensor,
            structure_tokens: torch.Tensor
    ):
        output = self.model.forward(
            structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens
        )
        # print(f"Forward time: {time.time()-t1}")
        sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)
        return sequence_tokens

    def decode(self,
               sequence_tokens):
        print (sequence_tokens[0].shape)
        sequence = self.tokenizer.decode(sequence_tokens[0])
        print(sequence)
        # print(f"Total time: {time.time()-t0}")
        return sequence
    
def warmup(model, coords, plddt, structure_tokens):
    with torch.no_grad():
        for i in range(5):
            sequence_tokens=model(coords, plddt, structure_tokens)
            
def time_run(model, coords, plddt, structure_tokens):
    sequence_tokens = None
    with torch.no_grad():
        sequence_tokens=model(coords, plddt, structure_tokens)
        model.decode(sequence_tokens)

def export_transformer(model, in1, in2):
    coords, plddt, structure_tokens = in1
    coords2, plddt2, structure_tokens2 = in2

    with torch.no_grad():
        inputs = model.model._forward_pre(structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens)
        inputs2 = model.model._forward_pre(structure_coords=coords2, per_res_plddt=plddt2, structure_tokens=structure_tokens2)
        from polygraphy.json import save_json
        x, sequence_id, affine_tensor, affine_mask, chain_id = inputs
        save_json([{'x':x, 'sequence_id': sequence_id, 'affine_tensor': affine_tensor, 'affine_mask': affine_mask, 'chain_id': chain_id}], 't_inputs.json')

        ex_wrapper = TransformerStackWrapper.wrap(model.model)
        trt_wrapper = TRTWrapper("onnx/trans_legacy", ex_wrapper)
        if trt_wrapper.has_onnx():
            return
        outputs = ex_wrapper(*inputs)
        start = time.time()
        outputs2 = ex_wrapper(*inputs2)
        stop = time.time()
        print(f"Pytorch run time: {stop-start} mean_std: ",
              [torch.std_mean(i.float()) for i in outputs2],
              "\nshapes:", [i.shape for i in outputs2]
        )
        
        dynamo = False

        trt_wrapper.onnx_export(inputs2,
                                dynamic_axes=ex_wrapper.dynamic_axes,
                                verbose=False)
        
        ort_cuda = True
        ort_sess = verification._ort_session(trt_wrapper.onnx_path,
                                             ort_providers=["CUDAExecutionProvider" if ort_cuda else "CPUExecutionProvider"])
        
        for i in range (5):
            ort_outs = verification._run_onnx(ort_sess, inputs)
        start = time.time()
        ort_outs = verification._run_onnx(ort_sess, inputs2)
        stop = time.time()
        print(f"ORT run time: {stop-start} mean_std: ", [torch.std_mean(torch.tensor(i).float()) for i in ort_outs])

        ver_opt = verification.VerificationOptions()
        # Parameters for runtime check:
        ver_opt.rtol = 0.001
        ver_opt.atol = 0.001
        check_ort_output=False
        if check_ort_output:
            verification._compare_onnx_pytorch_outputs(
                onnx_outs=ort_outs,
                pt_outs=outputs2,
                options=ver_opt,
            )
        return


def trt_transformer(model, in1, in2):
    coords, plddt, structure_tokens = in1
    coords2, plddt2, structure_tokens2 = in2

    with torch.no_grad():
        inputs = model.model._forward_pre(structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens)
        inputs2 = model.model._forward_pre(structure_coords=coords2, per_res_plddt=plddt2, structure_tokens=structure_tokens2)
        x, sequence_id, affine_tensor, affine_mask, chain_id = inputs2
        
        ex_wrapper = TransformerStackWrapper.wrap(model.model)
        trt_wrapper = TRTWrapper("onnx/trans_legacy", ex_wrapper)
        trt_wrapper.load_engine()
        outputs = ex_wrapper(*inputs)
        start = time.time()
        outputs2 = ex_wrapper(*inputs2)
        
        stop = time.time()
        print(f"Pytorch run time: {stop-start} mean_std: ",
              [torch.std_mean(i.float()) for i in outputs2],
              "\nshapes:", [i.shape for i in outputs2]
        )
        if not trt_wrapper.has_engine():
            trt_wrapper.build_engine(input_profiles=ex_wrapper.sample_profile(max_len=affine_mask.size(1)),
                                     tf32=True,
                                     bf16=True
            )
            trt_wrapper.engine.save()
        if trt_wrapper.has_engine():
            trt_inputs={"x":x, "sequence_id":sequence_id, "affine_tensor":affine_tensor, "affine_mask":affine_mask, "chain_id":chain_id}
            start = time.time()
            trt_outputs=trt_wrapper(**trt_inputs)
            stop = time.time()
            print(f"TRT run time: {stop-start} mean_std: ",
                  [torch.std_mean(i.float()) for i in trt_outputs],
                  "\nshapes:", [i.shape for i in trt_outputs]
            )
            check_trt_output=True
            if check_trt_output:
                torch.testing.assert_close(
                    trt_outputs,
                    outputs2,
                    rtol=0.01,
                    atol=0.01
                )
    
def get_coords(model, fname):
    chain = ProteinChain.from_pdb(fname)
    coords, plddt = get_structure_inputs_for_chain(chain)
    device = next(model.parameters()).device
    coords = coords.to(device=device)
    plddt = plddt.to(device=device)
    structure_tokens = model.encode(coords)
    return coords, plddt, structure_tokens

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('medium') # | 'high')
    os.makedirs("onnx", exist_ok=True)
    
    model = Invfold().cuda().eval()

    inp = get_coords(model, "esm/data/1utn.pdb")
    coords, plddt, structure_tokens = inp

    from polygraphy.json import save_json
    save_json([{'coords':coords, 'plddt': plddt, 'structure_tokens': structure_tokens}], 'inputs.json')
    structure_tokens=structure_tokens.cuda()
    
    warmup(model, coords, plddt, structure_tokens)
    time_run(model, coords, plddt, structure_tokens)
    # in2 = get_coords(model, "esm/data/1a0q.pdb")

    export_transformer(model, inp, inp) # in2)
    trt_transformer(model, inp, inp) # in2)
