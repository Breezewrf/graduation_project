# -*- coding: utf-8 -*-
# @Time    : 2022/3/5 17:03
# @Author  : Breeze
# @Email   : 578450426@qq.com
from network.blocks import *
from network.blocks import _upsample_like
from torch import sigmoid


class Encoder(nn.Module):
    # input: [1, 3, 128, 128]
    # out:
    #   out[0]:[1, 512, 8, 8]
    #   out[1]:[1, 32, 128, 128], [1, 64, 64, 64], [1, 128, 32, 32], [1, 256, 16, 16], [1, 512, 8, 8]
    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True):
        super(Encoder, self).__init__()
        self.down_convs = []
        self.RSU_5 = RSU6(3, 16, 32)
        self.RSU_4 = RSU6(32, 16, 64)
        self.RSU_3 = RSU5(64, 16, 128)
        self.RSU_2 = RSU5(128, 64, 256)
        self.RSU_1 = RSU4(256, 128, 512)

        # reset_params(self)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        迭代出来的x是整个encode的最终输出，encoder_outs是encode中间输出，用于跳跃连接
        """

        encoder_outs = []
        pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        print("RSU5")
        x = self.RSU_5(x)
        encoder_outs.append(x)
        x = pool(x)

        print("RSU4")
        x = self.RSU_4(x)
        encoder_outs.append(x)
        x = pool(x)

        print("RSU3")
        x = self.RSU_3(x)
        encoder_outs.append(x)
        x = pool(x)

        print("RSU2")
        x = self.RSU_2(x)
        encoder_outs.append(x)
        x = pool(x)

        print("RSU1")
        x = self.RSU_1(x)
        encoder_outs.append(x)
        x = pool(x)

        return x, encoder_outs


class image_Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, is_mask_decoder=True):
        super(image_Decoder, self).__init__()
        self.conv_final = None
        self.is_mask_decoder = is_mask_decoder
        self.up_convs = []
        self.RSU_1_d = RSU4(1024, 256, 512)
        self.RSU_2_d = RSU5(768, 128, 256)
        self.RSU_3_d = RSU5(384, 64, 128)
        self.RSU_4_d = RSU6(192, 16, 32)
        self.RSU_5_d = RSU6(64, 8, 16)

        self.side_image1 = nn.Conv2d(512, 3, 3, padding=1)
        self.side_image2 = nn.Conv2d(256, 3, 3, padding=1)
        self.side_image3 = nn.Conv2d(128, 3, 3, padding=1)
        self.side_image4 = nn.Conv2d(32, 3, 3, padding=1)
        self.side_image5 = nn.Conv2d(16, 3, 3, padding=1)
        self.outConv = nn.Conv2d(15, 3, 1)

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

    # 加入encode的跳跃连接

    def forward(self, x, encoder_outs=None):
        d_shape = torch.ones(1, 3, 512, 512)
        x = _upsample_like(x, encoder_outs[-1])
        x = self.RSU_1_d(torch.cat((x, encoder_outs[-1]), 1))
        d1 = self.side_image1(x)
        d1 = _upsample_like(d1, d_shape)
        """
        d1 => (1, 32, 32) => (1, 512, 512)
        """
        x = _upsample_like(x, encoder_outs[-2])
        x = self.RSU_2_d(torch.cat((x, encoder_outs[-2]), 1))
        d2 = self.side_image2(x)
        d2 = _upsample_like(d2, d_shape)

        x = _upsample_like(x, encoder_outs[-3])
        x = self.RSU_3_d(torch.cat((x, encoder_outs[-3]), 1))
        d3 = self.side_image3(x)
        d3 = _upsample_like(d3, d_shape)

        x = _upsample_like(x, encoder_outs[-4])
        x = self.RSU_4_d(torch.cat((x, encoder_outs[-4]), 1))
        d4 = self.side_image4(x)
        d4 = _upsample_like(d4, d_shape)

        x = _upsample_like(x, encoder_outs[-5])
        x = self.RSU_5_d(torch.cat((x, encoder_outs[-5]), 1))
        d5 = self.side_image5(x)
        d5 = _upsample_like(d5, d_shape)

        d0 = self.outConv(torch.cat((d1, d2, d3, d4, d5), 1))

        return sigmoid(d0), sigmoid(d1), sigmoid(d2), sigmoid(d3), sigmoid(d4), sigmoid(d5)


class mask_Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, is_mask_decoder=True):
        super(mask_Decoder, self).__init__()
        self.conv_final = None
        self.is_mask_decoder = is_mask_decoder
        self.up_convs = []
        self.RSU_1_d = RSU4(1024, 256, 512)
        self.RSU_2_d = RSU5(768, 128, 256)
        self.RSU_3_d = RSU5(384, 64, 128)
        self.RSU_4_d = RSU6(192, 16, 32)
        self.RSU_5_d = RSU6(64, 8, 16)
        self.side_mask1 = nn.Conv2d(512, 1, 3, padding=1)
        self.side_mask2 = nn.Conv2d(256, 1, 3, padding=1)
        self.side_mask3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side_mask4 = nn.Conv2d(32, 1, 3, padding=1)
        self.side_mask5 = nn.Conv2d(16, 1, 3, padding=1)
        self.outConv = nn.Conv2d(5, 1, 1)

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

    # 加入encode的跳跃连接
    def forward(self, x, encoder_outs=None):
        d_shape = torch.ones(1, 1, 512, 512)
        x = _upsample_like(x, encoder_outs[-1])
        x = self.RSU_1_d(torch.cat((x, encoder_outs[-1]), 1))
        d1 = self.side_mask1(x)
        d1 = _upsample_like(d1, d_shape)
        """
        d1 => (1, 32, 32) => (1, 512, 512)
        """
        x = _upsample_like(x, encoder_outs[-2])
        x = self.RSU_2_d(torch.cat((x, encoder_outs[-2]), 1))
        d2 = self.side_mask2(x)
        d2 = _upsample_like(d2, d_shape)

        x = _upsample_like(x, encoder_outs[-3])
        x = self.RSU_3_d(torch.cat((x, encoder_outs[-3]), 1))
        d3 = self.side_mask3(x)
        d3 = _upsample_like(d3, d_shape)

        x = _upsample_like(x, encoder_outs[-4])
        x = self.RSU_4_d(torch.cat((x, encoder_outs[-4]), 1))
        d4 = self.side_mask4(x)
        d4 = _upsample_like(d4, d_shape)

        x = _upsample_like(x, encoder_outs[-5])
        x = self.RSU_5_d(torch.cat((x, encoder_outs[-5]), 1))
        d5 = self.side_mask5(x)
        d5 = _upsample_like(d5, d_shape)

        d0 = self.outConv(torch.cat((d1, d2, d3, d4, d5), 1))

        return sigmoid(d0), sigmoid(d1), sigmoid(d2), sigmoid(d3), sigmoid(d4), sigmoid(d5)
