import torch

class TransformerStackWrapper(torch.nn.Module):
    """
    An auxiliary class to facilitate ONNX->TRT export of transformer stack
    """
    def __init__(self, model, precision="fp32"):
        super().__init__()
        self.input_names = ['x', 'sequence_id', 'affine_tensor', 'affine_mask', 'chain_id']
        self.output_names = ['x_out', 'embedding']
        self.dynamic_axes = {'x':{0:"batch", 1:"len"},
                             'sequence_id': {0:"batch", 1:"len"},
                             'affine_tensor': {0:"batch", 1:"len"},
                             'affine_mask': {0:"batch", 1:"len"},
                             'chain_id': {0:"batch", 1:"len"}}
        self.transformer = model.transformer
        self.precision = precision
        if self.precision=="bf16":
            print("Converting transformer stack to BF16 ...")
        
    def get_export_obj(self):
        return self
        
    def sample_profile(self, min_len=16, max_len=225):
        return [{'x': [[1, min_len, 1536], [1, max_len, 1536], [1, max_len, 1536]],
                 'sequence_id': [[1, min_len], [1, max_len], [1, max_len]],
                 'affine_tensor': [[1, min_len, 12], [1, max_len, 12], [1, max_len, 12]],
                 'affine_mask': [[1, min_len], [1, max_len], [1, max_len]],
                 'chain_id': [[1, min_len], [1, max_len], [1, max_len]],
                 }]

    def can_handle(self, x, sequence_id, affine_tensor, affine_mask, chain_id ):
        return x.shape[1] <= 1700

    def forward(self, x, sequence_id, affine_tensor, affine_mask, chain_id ):
        # if self.precision=="bf16":
        x_out, embedding  = self.transformer(x, sequence_id, affine_tensor, affine_mask, chain_id)
        # if self.precision=="bf16":
        return x_out, embedding

    @classmethod
    def wrap(cls, model, precision="float32"):
        wrapper = cls(model, precision)
        return wrapper

