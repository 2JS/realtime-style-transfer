import argparse
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.models.neural_network.quantization_utils import quantize_weights

import net
from function import adaptive_instance_normalization as adain

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg = net.vgg
        self.net = nn.Sequential(*list(self.vgg.children())[2:31])

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = net.decoder

    def forward(self, content, style):
        out = adain(content, style)
        y = self.net(out)
        _, _, h, w = y.shape
        # return y.permute(0, 2, 3, 1)
        return y




# decoder.load_state_dict(torch.load('models/decoder.pth'))
# vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
# vgg = nn.Sequential(*list(vgg.children())[:31])

encoder = Encoder()
encoder.vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
# encoder.net[0].weight = nn.Parameter(encoder.net[0].weight * 255)
# encoder = encoder.to(memory_format=torch.channels_last)

decoder = Decoder()
decoder.net.load_state_dict(torch.load('models/decoder.pth'))
# decoder = decoder.to(memory_format=torch.channels_last)

sample_input = torch.rand(1, 3, 640, 480)
traced_vgg = torch.jit.trace(encoder, sample_input)
latent = traced_vgg(sample_input)
converted_vgg  = ct.convert(
    traced_vgg,
    source='pytorch',
    # inputs = [ct.TensorType(shape=sample_input.shape)]
    inputs = [ct.ImageType(shape=sample_input.shape, scale=255, bias=[-103.9390, -116.7790, -123.6800], color_layout=ct.colorlayout.BGR)]
)
# converted_vgg = quantize_weights(converted_vgg, nbits=16)

converted_vgg.save("adain_vgg.mlmodel")

sample_latent = torch.rand_like(latent)
traced_decoder = torch.jit.trace(decoder, (sample_latent, sample_latent))
out = traced_decoder(sample_latent, sample_latent)
converted_decoder = ct.convert(
    traced_decoder,
    source='pytorch',
    inputs = [ct.TensorType(shape=sample_latent.shape), ct.TensorType(shape=sample_latent.shape)],
    outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)]
)
# converted_decoder = quantize_weights(converted_decoder, nbits=16)
converted_decoder.save("adain_dec.mlmodel")
