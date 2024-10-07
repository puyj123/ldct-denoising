import os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg19

class WGAN_VGG(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, input_size=64):
        super(WGAN_VGG, self).__init__()
        #self.generator = WGAN_VGG_generator()
        #self.discriminator = WGAN_VGG_discriminator(input_size)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss()

    def p_loss(self, x, y):
        fake = x.repeat(1, 3, 1, 1)
        real = y.repeat(1, 3, 1, 1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss



class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature_extractor(x)
        return out
