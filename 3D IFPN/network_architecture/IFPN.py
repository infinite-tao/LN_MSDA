from torch import nn
import torch
import torch.nn.functional as F
from network_architecture.BackBone3D import BackBone3D
# from network_architecture.neural_network import SegmentationNetwork
from network_architecture.FDTblock import SAT_CAT


class IFPN3D(nn.Module):
    def __init__(self, vis=False):
        super(IFPN3D, self).__init__()
        # self.training = train
        self.backbone = BackBone3D()

        self.down4 = nn.Sequential(
            nn.Conv3d(1024, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU()
        )

        self.SCAT = SAT_CAT(64, 64, 3)

        self.fuse1 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.predict = nn.Conv3d(64, 2, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(192, 2)

        self.avgpool1 = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(64, 2)

        self.avgpool2 = nn.AdaptiveAvgPool3d(1)
        self.fc2 = nn.Linear(64, 2)
    def forward(self, x):
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        # Top-down
        down4 = self.down4(layer4)
        down3 = torch.add(
            F.upsample(down4, size=layer3.size()[2:], mode='trilinear'),
            self.down3(layer3)
        )
        down2 = torch.add(
            F.upsample(down3, size=layer2.size()[2:], mode='trilinear'),
            self.down2(layer2)
        )
        down1 = torch.add(
            F.upsample(down2, size=layer1.size()[2:], mode='trilinear'),
            self.down1(layer1)
        )

        SCAT1 = self.SCAT(down1, down2)
        SCAT2 = self.SCAT(SCAT1, down3)

        feature_l1 = self.avgpool(SCAT1)
        feature_l1 = feature_l1.view(feature_l1.size(0), -1)
        feature_l2 = self.avgpool(SCAT2)
        feature_l2 = feature_l2.view(feature_l2.size(0), -1)
        feature_l3 = self.avgpool(down4)
        feature_l3 = feature_l3.view(feature_l3.size(0), -1)

        fusion_class = torch.cat((feature_l1, feature_l2, feature_l3), 1)
        final_class = self.fc(fusion_class)

        feature_gender = self.avgpool1(down4)
        feature_gender = feature_gender.view(feature_gender.size(0), -1)
        final_gender = self.fc1(feature_gender)

        feature_age = self.avgpool2(down4)
        feature_age = feature_age.view(feature_age.size(0), -1)
        final_age = self.fc2(feature_age)

        down40 = F.upsample(down4, size=layer1.size()[2:], mode='trilinear')
        down30 = F.upsample(down3, size=layer1.size()[2:], mode='trilinear')
        down20 = F.upsample(down2, size=layer1.size()[2:], mode='trilinear')
        down10 = F.upsample(down1, size=layer1.size()[2:], mode='trilinear')

        fusion = self.fuse1(torch.cat((down40, down30, down20, down10), 1))
        fusion = F.upsample(fusion, size=x.size()[2:], mode='trilinear')
        predict = self.predict(fusion)

        return predict, final_class, final_gender, final_age

