import torch
import coremltools as ct
from dummy.dummy import Dummy


model = Dummy()
model.eval()

sample_input = (torch.rand(1, 4, 256, 256), torch.rand(1, 4, 256, 256))
traced_model = torch.jit.trace(model, sample_input)
out = traced_model(*sample_input)

converted_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=sample_input[0].shape), ct.TensorType(shape=sample_input[1].shape)]
)

converted_model.save("Dummy.mlmodel")