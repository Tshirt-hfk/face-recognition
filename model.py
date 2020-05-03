import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义make_layer
def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),
        nn.BatchNorm2d(out_channel))
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)


# 定义ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


# 堆叠Resnet
class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(3, 2, 1))
        self.layer1 = make_layer(64, 64, 2)
        self.layer2 = make_layer(64, 128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 512, 2, stride=2)
        self.avg = nn.AvgPool2d(7)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return x


class FaceRecognition(nn.Module):
    def __init__(self, label_num=394):
        super(FaceRecognition, self).__init__()
        self.res = Resnet()
        self.classifier = nn.Sequential(nn.Linear(512, label_num))

    def forward(self, x):
        x = self.res(x)
        x = self.classifier(x)
        return x
