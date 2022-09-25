from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=1,
            groups=4,
        )

    def forward(self, first, second):
        return self.conv(first) + self.conv(second)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters, lr=1e-3)
        return optim
