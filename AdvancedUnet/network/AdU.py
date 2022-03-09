# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 16:59
# @Author  : Breeze
# @Email   : 578450426@qq.com
from torch import nn
from network.blocks import *
from network.en_decoders import *
import torch


class AdUNet(nn.Module):
    """
    input: (batch, 3, 512, 512)
    """

    def __init__(self, n_channels=3, bilinear=False, transfer_data=True):
        super(AdUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.transfer_data = transfer_data

        self.Encoder = Encoder()
        self.mask_Decoder = mask_Decoder()
        self.image_Decoder = image_Decoder()
        self.optimizer_encoder = None
        self.optimizer_image = None
        self.optimizer_mask = None

    def set_optimizers(self):
        self.optimizer_encoder = torch.optim.Adam(self.Encoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                                  weight_decay=0)
        self.optimizer_image = torch.optim.Adam(self.image_Decoder.parameters(), lr=0.001, betas=(0.9, 0.999),
                                                eps=1e-08, weight_decay=0)
        self.optimizer_mask = torch.optim.Adam(self.mask_Decoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                               weight_decay=0)

    def step_all(self):
        self.optimizer_encoder.step()
        self.optimizer_image.step()
        self.optimizer_mask.step()

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_image.zero_grad()
        self.optimizer_mask.zero_grad()
        if self.vm_decoder is not None:
            self.optimizer_vm.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def forward(self, x):
        x1, encoder_outs = self.Encoder(x)
        masks = self.mask_Decoder(x1, encoder_outs)
        images = self.image_Decoder(x1, encoder_outs)
        return masks, images


if __name__ == '__main__':
    net = AdUNet()
    im = torch.randn(1, 3, 512, 512)
    t = net(im)
    print("testing")
