import torch
import torch.nn as nn
from typing import Tuple, List


class BasicModel_c(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        dim_imag = 32 # Dimension of the images in input
        
        kern_conv = 3
        stride_conv = 1
        pad_conv = 1
        
        kern_pool = 2
        stride_pool= 2
        # Define the convolutional layers
        
        self.init_layer = nn.Sequential(
            nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=32,
                    kernel_size=kern_conv,
                    stride=stride_conv,
                    padding=pad_conv
                ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(
                    kernel_size=kern_pool,
                    stride=stride_pool
                ),
            ###
            nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=kern_conv,
                    stride=stride_conv,
                    padding=pad_conv
                ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(
                    kernel_size=kern_pool,
                    stride=stride_pool
                ),
            
            ###
            nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=kern_conv,
                    stride=stride_conv,
                    padding=pad_conv
                ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                    in_channels=64,
                    out_channels=output_channels[0],
                    kernel_size=kern_conv,
                    stride=2,
                    padding=pad_conv
                ),
            nn.BatchNorm2d(output_channels[0]),
            nn.ReLU()
        )
        
        self.middle_layer1 = nn.Sequential(
                    #nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[0],
                            out_channels=output_channels[1],
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[1]),
                    nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[1],
                            out_channels=output_channels[1],
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[1]),
                    nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[1],
                            out_channels=output_channels[1],
                            kernel_size=3,
                            stride=2,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[1]),
                    nn.ReLU()
                    
                    
        )
        
        self.middle_layer2 = nn.Sequential(
                    #nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[1],
                            out_channels=output_channels[2],
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[2]),
                    nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[2],
                            out_channels=output_channels[2],
                            kernel_size=3,
                            stride=2,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[2]),
                    nn.ReLU()
        )
        self.middle_layer3 = nn.Sequential(
                    #nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[2],
                            out_channels=output_channels[3],
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[3]),
                    nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[3],
                            out_channels=output_channels[3],
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[3]),
                    nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[3],
                            out_channels=output_channels[3],
                            kernel_size=3,
                            stride=2,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[3]),
                    nn.ReLU()
        )
        self.middle_layer4 = nn.Sequential(
                    #nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[3],
                            out_channels=output_channels[4],
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[4]),
                    nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[4],
                            out_channels=output_channels[4],
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[4]),
                    nn.ReLU(),
                    nn.Conv2d(
                            in_channels=output_channels[4],
                            out_channels=output_channels[4],
                            kernel_size=3,
                            stride=2,
                            padding=1
                        ),
                    nn.BatchNorm2d(output_channels[4]),
                    nn.ReLU()
                    
        )    
        self.last_layer = nn.Sequential(
            #nn.ReLU(),
            nn.Conv2d(
                    in_channels=output_channels[4],
                    out_channels=output_channels[5],
                    kernel_size=kern_conv,
                    stride=1,
                    padding=pad_conv
                ),
            #nn.BatchNorm2d(output_channels[5]),
            nn.ReLU(),
            nn.Conv2d(
                    in_channels=output_channels[5],
                    out_channels=output_channels[5],
                    kernel_size=3,
                    stride=1,
                    padding=0
                ),
            #nn.BatchNorm2d(output_channels[5]),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        
        batch_size = x.shape[0]
        out_features = []
        
        out_features.append(self.init_layer(x))
        out_features.append(self.middle_layer1(out_features[0]))
        out_features.append(self.middle_layer2(out_features[1]))
        out_features.append(self.middle_layer3(out_features[2]))
        out_features.append(self.middle_layer4(out_features[3]))
        out_features.append(self.last_layer(out_features[4]))
        
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)