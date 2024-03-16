import torch
import torch.nn as nn
from torchvision import models



def get_spatial_extractor(spatial_extractor, get_feats=False):
    if spatial_extractor == 'resnet50':
        model = ResNet(get_feats, layers=50)
    elif spatial_extractor == 'densenet':
        model = DenseNet121(get_feats)
    elif spatial_extractor == 'googlenet':
        model = GoogLeNet(get_feats)
    elif spatial_extractor == 'vgg':
        model = VGG16(get_feats)
    else:
        model = ResNet(get_feats, layers=18)
    return model


class ResNet(torch.nn.Module):
    def __init__(self, get_feats, layers=18):
        super(ResNet, self).__init__()
        if layers == 18:
            resnet = models.resnet18(pretrained=True)
            self.share = torch.nn.Sequential()
            self.share.add_module("conv1", resnet.conv1)
            self.share.add_module("bn1", resnet.bn1)
            self.share.add_module("relu", resnet.relu)
            self.share.add_module("maxpool", resnet.maxpool)
            self.share.add_module("layer1", resnet.layer1)
            self.share.add_module("layer2", resnet.layer2)
            self.share.add_module("layer3", resnet.layer3)
            self.share.add_module("avgpool", resnet.avgpool)
            self.share.add_module("reshape", nn.Conv2d(256, 512, kernel_size=3, padding=1))
        else:
            resnet = models.resnet50(pretrained=True)
            self.share = torch.nn.Sequential()
            self.share.add_module("conv1", resnet.conv1)
            self.share.add_module("bn1", resnet.bn1)
            self.share.add_module("relu", resnet.relu)
            self.share.add_module("maxpool", resnet.maxpool)
            self.share.add_module("layer1", resnet.layer1)
            self.share.add_module("layer2", resnet.layer2)
            self.share.add_module("layer3", resnet.layer3)
            self.share.add_module("layer4", resnet.layer4)
            self.share.add_module("avgpool", resnet.avgpool)
            self.share.add_module("reshape", nn.Conv2d(2048, 512, kernel_size=3, padding=1))
        self.fc = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 5)
            )
        self.get_feats = get_feats

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 512)
        if self.get_feats:
            feats = x
            y = self.fc(x)
            return y, feats
        else:
            return x
    

class GoogLeNet(torch.nn.Module):
    def __init__(self, get_feats):
        super(GoogLeNet, self).__init__()
        googlenet = models.googlenet(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", googlenet.conv1)
        self.share.add_module("maxpool1", googlenet.maxpool1)
        self.share.add_module("conv2", googlenet.conv2)
        self.share.add_module("conv3", googlenet.conv3)
        self.share.add_module("maxpool2", googlenet.maxpool2)
        self.share.add_module("inception3a", googlenet.inception3a)
        self.share.add_module("inception3b", googlenet.inception3b)
        self.share.add_module("maxpool3", googlenet.maxpool3)
        self.share.add_module("inception4a", googlenet.inception4a)
        self.share.add_module("inception4b", googlenet.inception4b)
        self.share.add_module("inception4c", googlenet.inception4c)
        self.share.add_module("inception4d", googlenet.inception4d)
        self.share.add_module("inception4e", googlenet.inception4e)
        self.share.add_module("maxpool4", googlenet.maxpool4)
        self.share.add_module("inception5a", googlenet.inception5a)
        self.share.add_module("inception5b", googlenet.inception5b)
        self.share.add_module("avgpool", googlenet.avgpool)
        self.share.add_module("dropout", googlenet.dropout)
        self.share.add_module("reshape", nn.Conv2d(1024, 512, kernel_size=3, padding=1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 5)
        )
        self.get_feats = get_feats

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 512)
        if self.get_feats:
            feats = x
            y = self.fc(x)
            return y, feats
        else:
            return x


class VGG16(torch.nn.Module):
    def __init__(self, get_feats):
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("features", vgg.features)
        self.share.add_module("avgpool", nn.AdaptiveAvgPool2d(output_size=(1,1)))
        #self.share.add_module("reshape", nn.Conv2d(512, 2048, kernel_size=3, padding=1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 5)
        )
        self.get_feats = get_feats

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 512)
        if self.get_feats:
            feats = x
            y = self.fc(x)
            return y, feats
        else:
            return x
    

class DenseNet121(torch.nn.Module):
    def __init__(self, get_feats):
        super(DenseNet121, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("features", densenet.features)
        self.share.add_module("reshape", nn.Conv2d(1024, 512, kernel_size=3, padding=1)) ###
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 5)
        )
        self.get_feats = get_feats

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 512)
        if self.get_feats:
            feats = x
            y = self.fc(x)
            return y, feats
        else:
            return x