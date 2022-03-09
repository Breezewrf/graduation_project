# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 9:34
# @Author  : Breeze
# @Email   : 578450426@qq.com
import torch
import torchvision
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import os
import numpy as np
import glob

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction='mean')


def mul_bce_loss_fusion(d0, d1, d2, d3, d4, d5, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data[0], loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0], loss5.data[0], loss6.data[0]))

    return loss0, loss


def l1_relative(reconstructed, real, batch, area):
    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1
