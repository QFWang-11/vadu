from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
from detectron2.layers import FrozenBatchNorm2d
import copy

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],    # res1 一个卷积层
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 256  #ResNet34
        #self.out_channels = 1024  #ResNet50

        # Convert BatchNorm layers to FrozenBatchNorm2d
        #FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])

#构建ResNet的第五层
class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [256, 512]
        #self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        #x = F.adaptive_avg_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        #feat = F.adaptive_avg_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])

# Res5Head initializes itself as an nn.Sequential module containing resnet.layer4 (the 5th layer of ResNet).
# After initialization, self[0] refers to resnet.layer4, which is an nn.Sequential module containing a sequence of ResNet blocks
#class Res5HeadLastStride1(Res5Head):
#    def __init__(self, resnet):
#        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
#        self[0].conv2.stride=(1,1)
#        self[0].downsample.stride=(1,1)

class Res5HeadLastStride1(Res5Head):
    def __init__(self, resnet):
        super(Res5HeadLastStride1, self).__init__(resnet)
        
        # 获取layer4模块
        layer4 = self[0]
        
        # 只修改第一个block（通常只有第一个block负责下采样）
        first_block = layer4[0]
        
        # 处理BasicBlock（ResNet18/34）
        if isinstance(first_block, torchvision.models.resnet.BasicBlock):
            # 修改conv1的stride（BasicBlock中下采样由conv1控制）
            first_block.conv1.stride = (1, 1)
            if first_block.downsample is not None:
                for m in first_block.downsample.modules():
                    if isinstance(m, nn.Conv2d):
                        m.stride = (1, 1)
        
        # 处理Bottleneck（ResNet50/101/152）
        elif isinstance(first_block, torchvision.models.resnet.Bottleneck):
            # 修改conv2的stride（Bottleneck中下采样由conv2控制）
            first_block.conv2.stride = (1, 1)
            if first_block.downsample is not None:
                for m in first_block.downsample.modules():
                    if isinstance(m, nn.Conv2d):
                        m.stride = (1, 1)
    
#构建ResNet模型，返回Backbone和Res5Head对象
def build_resnet(name="resnet34", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)

def build_resnet_last_stride1(name="resnet34", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet),Res5HeadLastStride1(copy.deepcopy(resnet)) # NOTE backbone, detection res5, reid res5
