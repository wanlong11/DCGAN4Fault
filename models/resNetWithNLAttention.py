import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
        else:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )

        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class ResNet18NonLocal(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18NonLocal, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)
        self.non_local1 = NonLocalBlock(in_channels=64)
        self.non_local2 = NonLocalBlock(in_channels=128)
        self.non_local3 = NonLocalBlock(in_channels=256)
        self.non_local4 = NonLocalBlock(in_channels=512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.non_local1(x)
        x = self.resnet18.layer2(x)
        x = self.non_local2(x)
        x = self.resnet18.layer3(x)
        x = self.non_local3(x)
        x = self.resnet18.layer4(x)
        x = self.non_local4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50NonLocal(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50NonLocal, self).__init__()

        self.resnet50 = models.resnet50(pretrained=False)
        self.non_local1 = NonLocalBlock(in_channels=256)
        self.non_local2 = NonLocalBlock(in_channels=512)
        self.non_local3 = NonLocalBlock(in_channels=1024)
        self.non_local4 = NonLocalBlock(in_channels=2048)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.non_local1(x)
        x = self.resnet50.layer2(x)
        x = self.non_local2(x)
        x = self.resnet50.layer3(x)
        x = self.non_local3(x)
        x = self.resnet50.layer4(x)
        x = self.non_local4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x