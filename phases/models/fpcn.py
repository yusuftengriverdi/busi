import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .resnet import resnet18_features
from .layers import conv1x1_bn, conv3x3_bn
import numpy as  np

# Source: https://github.com/c0nn3r/RetinaNet/blob/master/resnet_features.py 
class FeaturePyramid(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_3 = conv1x1_bn(128, 256)
        self.pyramid_transformation_4 = conv1x1_bn(256, 256)
        self.pyramid_transformation_5 = conv1x1_bn(512, 256)

        # both based around resnet_feature_5
        self.pyramid_transformation_6 = conv3x3_bn(512, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3_bn(256, 256, padding=1, stride=2)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3_bn(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3_bn(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        height, width = scaled_feature.size()[2:]
        return F.interpolate(original_feature, scale_factor=scale_factor, mode='bilinear', align_corners = True)[:, :, :height, :width]

    def forward(self, x):

        # don't need resnet_feature_2 as it is too large
        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)

        pyramid_feature_4 = self.upsample_transform_1(
            torch.add(upsampled_feature_5, pyramid_feature_4)
        )

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)

        pyramid_feature_3 = self.upsample_transform_2(
            torch.add(upsampled_feature_4, pyramid_feature_3)
        )

        return (pyramid_feature_3,
                pyramid_feature_4,
                pyramid_feature_5,
                pyramid_feature_6,
                pyramid_feature_7)


class SubNet(nn.Module):

    def __init__(self, mode, num_classes=2, depth=3,
                 base_activation=F.relu,
                 output_activation=F.sigmoid):
        super(SubNet, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.base_activation = base_activation
        self.output_activation = output_activation
        self.sum_ = 0
        self.subnet_base = nn.ModuleList([conv3x3_bn(1280, 1280, padding=1)
                                          for _ in range(depth)])


        if mode == 'classes':
            self.subnet_output = nn.Sequential( conv3x3_bn(1280, 256, padding=1),
                                                # nn.AdaptiveAvgPool2d(4),
                                                # Add a flatten layer to convert the tensor to 1D.
                                                nn.Flatten(),  
                                                nn.Linear(12544, 2048),  
                                                # nn.BatchNorm2d(2048),
                                                nn.Linear(2048, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, self.num_classes)
                                            )
                                                        

    def forward(self, x):
        for layer in self.subnet_base:
            x = self.base_activation(layer(x))

        x = self.subnet_output(x)
        x = self.output_activation(x)

        return x


class FPCN(nn.Module):

    def __init__(self, num_classes, backbone=resnet18_features, use_pretrained=False, device='cuda'):
        super(FPCN, self).__init__()
        self.num_classes = num_classes

        _resnet = backbone(pretrained=use_pretrained)
        self.feature_pyramid = FeaturePyramid(_resnet).to(device)

        # self.subnet_boxes = SubNet(mode='boxes', num_classes=self.num_classes)
        self.subnet_classes = SubNet(mode='classes', num_classes=self.num_classes)

        self.device = device

    def forward(self, x):
        classes = []

        features = self.feature_pyramid(x)
        
        features = [feature.to(self.device) for feature in features]
        # print([feature.shape for feature in features])

        downsample = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1).to(self.device)
        upsample = self.feature_pyramid._upsample

        downsample_0_0 = downsample(features[0])
        downsample_0_f = downsample(downsample_0_0)
        downsample_1_f = downsample(features[1])
        bootleneck = features[2]
        upsample_0_f = upsample(features[3], bootleneck)
        upsample_1_0 = upsample(features[-1], bootleneck)
        upsample_1_f = upsample(upsample_1_0, bootleneck)

        transformed_features = [downsample_0_f, downsample_1_f, bootleneck, upsample_0_f, upsample_1_f]
        # print([feature.shape for feature in transformed_features])

        concatenated_features = torch.cat(transformed_features, dim=1)  # Concatenate the features along the channel dimension
        # Element-wise multiplication
        # multiplied_features = torch.prod(concatenated_features, dim=1)

        # Pass the multiplied features through the subnet
        classes = self.subnet_classes(concatenated_features)
        # print(classes.shape)
        return classes


if __name__ == '__main__':

    import torchvision.datasets as dset

    net = FPCN(num_classes=10)
    # For first time downloading.
    # cifar10 = dset.CIFAR10("data/cifar10/", download=True)
    cifar10 = dset.CIFAR10("data/cifar10/", download=True)
    print(cifar10.data.shape)

    x  = torch.tensor(cifar10.data[:100]) / 255
    x = x.permute(0, 3, 1, 2)
    
    predictions = net(Variable(x))

    print(predictions.size(), predictions[0])

