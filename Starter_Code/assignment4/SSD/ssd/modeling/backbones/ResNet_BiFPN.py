import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, List, OrderedDict
from .BiFPN import BiFPN

def block1(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
                    nn.Conv2d(
                            in_channels=in_ch,
                            out_channels=512,
                            kernel_size=1,
                            stride=1,
                            padding=0
                        ),
                    nn.BatchNorm2d(512),
                    nn.Conv2d(
                            in_channels=512,
                            out_channels=512,
                            kernel_size=3,
                            stride=2,
                            padding=1
                        ),
                    nn.BatchNorm2d(512),
                    nn.Conv2d(
                            in_channels=512,
                            out_channels=out_ch,
                            kernel_size=1,
                            stride=1,
                            padding=0
                        ),
                    nn.ReLU()
            )

def block2(in_ch: int) -> nn.Sequential:
    return nn.Sequential(
                    nn.Conv2d(
                            in_channels=in_ch,
                            out_channels=512,
                            kernel_size=1,
                            stride=1,
                            padding=0
                        ),
                    nn.BatchNorm2d(512),
                    nn.Conv2d(
                            in_channels=512,
                            out_channels=512,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                    nn.BatchNorm2d(512),
                    nn.Conv2d(
                            in_channels=512,
                            out_channels=in_ch,
                            kernel_size=1,
                            stride=1,
                            padding=0
                        ),
                    nn.ReLU()
            )

pretrained_model = models.resnet101(pretrained=True)


"""
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
"""

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)
    
class Res_BiFPN(torch.nn.Module):
    
    """
    This is the ResNet 101 modified
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, 256, 32, 256),
     shape(-1, 512, 16, 128),
     shape(-1, 1024, 8, 64),
     shape(-1, 2048, 4, 32),
     shape(-1, 2048, 2, 16),
     shape(-1, 2048, 1, 8)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]],
            pretrained_model = pretrained_model):
        super(Res_BiFPN, self).__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        # I use the layers and weights of the ResNet101
        self.pre_model = pretrained_model
        
        # Define the convolutional layer
        self.layer5 = nn.Sequential(
                 block1(2048, 2048),
                 block2(2048),
                 block2(2048)
        )
        
        self.layer6 = nn.Sequential(
                 block1(2048, 2048),
                 block2(2048),
                 block2(2048)
        )
        self.conv_channel = [256,512,1024,2048,2048,2048]
        
        self.bifpn = nn.Sequential(
            *[BiFPN(200,
                    self.conv_channel,
                    True if _ == 0 else False)
                for _ in range(3)])
        
        #init_weights(self.layer5)
        #init_weights(self.layer6)
    
    

    def forward(self, x):
        
        """
        The forward functiom should output features with shape:
            [shape(-1, 256, 32, 256),
             shape(-1, 512, 16, 128),
             shape(-1, 1024, 8, 64),
             shape(-1, 2048, 4, 32),
             shape(-1, 2048, 2, 16),
             shape(-1, 2048, 1, 8)]
        """
        
        
        # ResNet w\o BiFPN
        
        out_features = []
        
        # layer 0 
        x = self.pre_model.conv1(x)
        x = self.pre_model.bn1(x)
        x = self.pre_model.relu(x)
        x = self.pre_model.maxpool(x)
        out_features.append(self.pre_model.layer1(x))
        #print(torch.size(out_features[0]))ss
        out_features.append(self.pre_model.layer2(out_features[0]))
        out_features.append(self.pre_model.layer3(out_features[1]))
        out_features.append(self.pre_model.layer4(out_features[2]))
        out_features.append(self.layer5(out_features[3]))
        out_features.append(self.layer6(out_features[4]))
        
        """
        
        # ResNet with BiFPN
        pre_features = OrderedDict()
        
        # layer 0 
        x = self.pre_model.conv1(x)
        x = self.pre_model.bn1(x)
        x = self.pre_model.relu(x)
        x = self.pre_model.maxpool(x)
        
        pre_features['feat0'] = self.pre_model.layer1(x)
        pre_features['feat1'] = self.pre_model.layer2(pre_features['feat0'])
        pre_features['feat2'] = self.pre_model.layer3(pre_features['feat1'])
        pre_features['feat3'] = self.pre_model.layer4(pre_features['feat2'])
        pre_features['feat4'] = self.layer5(pre_features['feat3'])
        pre_features['feat5'] = self.layer6(pre_features['feat4'])
        
        out_features = pre_features

        #print("model is in cuda: ", next(self.parameters()).is_cuda)
        """
        out_features = self.bifpn(out_features)
        
        #print([(k, v.shape) for k, v in out_features.items()])
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

