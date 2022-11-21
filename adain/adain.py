import argparse
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct

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
        self.net = nn.Sequential(*list(self.vgg.children())[:31])

    def forward(self, x):
        x = x[..., :3].permute(0, 3, 1, 2)
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = net.decoder

    def forward(self, content, style):
        out = adain(content, style)
        y = self.net(out)
        _, _, h, w = y.shape
        return torch.cat([y, torch.zeros(1, 1, h, w)], axis=1).permute(0, 2, 3, 1)




# decoder.load_state_dict(torch.load('models/decoder.pth'))
# vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
# vgg = nn.Sequential(*list(vgg.children())[:31])

encoder = Encoder()
encoder.vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
# encoder = encoder.to(memory_format=torch.channels_last)

decoder = Decoder()
decoder.net.load_state_dict(torch.load('models/decoder.pth'))
# decoder = decoder.to(memory_format=torch.channels_last)

sample_input = torch.rand(1, 640, 480, 4)
traced_vgg = torch.jit.trace(encoder, sample_input)
latent = traced_vgg(sample_input)
converted_vgg  = ct.convert(
    traced_vgg,
    source='pytorch',
    inputs = [ct.TensorType(shape=sample_input.shape)]
)

converted_vgg.save("adain_vgg.mlmodel")

sample_latent = torch.rand_like(latent)
traced_decoder = torch.jit.trace(decoder, (sample_latent, sample_latent))
out = traced_decoder(sample_latent, sample_latent)
converted_decoder = ct.convert(
    traced_decoder,
    source='pytorch',
    inputs = [ct.TensorType(shape=sample_latent.shape), ct.TensorType(shape=sample_latent.shape)]
)
converted_decoder.save("adain_dec.mlmodel")
