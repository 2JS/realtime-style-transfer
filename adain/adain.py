import argparse
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct

import net
from function import adaptive_instance_normalization, coral

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load('models/decoder.pth'))
vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:31])

sample_input = torch.rand(1, 3, 256, 256)
traced_vgg = torch.jit.trace(vgg, sample_input)
out = traced_vgg(sample_input)
converted_vgg  = ct.convert(
    traced_vgg,
    source='pytorch',
    inputs = [ct.TensorType(shape=sample_input.size())]
)

sample_latent = torch.rand(1, 512, 258, 258)
traced_decoder = torch.jit.trace(decoder, sample_latent)
out = traced_decoder(sample_latent)
converted_decoder = ct.convert(
    traced_decoder,
    source='pytorch',
    inputs = [ct.TensorType(shape=sample_latent.size())]
)
converted_decoder.save("adain_dec.mlmodel")
converted_vgg.save("adain_vgg.mlmodel")
