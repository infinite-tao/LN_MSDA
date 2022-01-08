import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Feature_Aggregator1(nn.Module):
    def __init__(self):
        super(Feature_Aggregator1, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.feature = nn.Linear(64, 64)
        self.FC = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.transpose(1, 3)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        y = self.FC(x)
        return x, y

    def get_parameters(self):
        parameter_list = [
            {'params': self.conv_layer1.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.conv_layer2.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.conv_layer3.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.feature.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {'params': self.FC.parameters(), 'lr_mult': 10, 'decay_mult': 2}
        ]
        return parameter_list


class Feature_Aggregator2(nn.Module):
    def __init__(self):
        super(Feature_Aggregator2, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.feature = nn.Linear(64, 64)
        self.FC = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.transpose(1, 3)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        y = self.FC(x)
        return x, y

    def get_parameters(self):
        parameter_list = [
            {'params': self.conv_layer1.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.conv_layer2.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.conv_layer3.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.feature.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {'params': self.FC.parameters(), 'lr_mult': 10, 'decay_mult': 2}
        ]
        return parameter_list


class Feature_Aggregator3(nn.Module):
    def __init__(self):
        super(Feature_Aggregator3, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1), nn.BatchNorm2d(1), nn.LeakyReLU(inplace=True))
        self.feature = nn.Linear(64, 64)
        self.FC = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.transpose(1, 3)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        y = self.FC(x)
        return x, y

    def get_parameters(self):
        parameter_list = [
            {'params': self.conv_layer1.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.conv_layer2.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.conv_layer3.parameters(), 'lr_mult': 1, 'decay_mult': 2},
            {'params': self.feature.parameters(), 'lr_mult': 10, 'decay_mult': 2},
            {'params': self.FC.parameters(), 'lr_mult': 10, 'decay_mult': 2}
        ]
        return parameter_list
