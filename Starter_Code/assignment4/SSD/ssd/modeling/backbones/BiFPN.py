import torch
import torchvision
import torch.nn as nn

def swish(x: torch.tensor) -> torch.tensor:
            return x * torch.sigmoid(x) 
    
class BiFPN(nn.Module):

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True,
                 use_p8=True):
        """
        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8

        # Conv layers
        self.conv7_up = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv6_up = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv5_up = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv4_up = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv3_up = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv4_down = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv5_down = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv6_down = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv7_down = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))
        self.conv8_down = nn.Sequential(nn.Conv2d(num_channels,num_channels,kernel_size=3,groups=num_channels,padding=1,bias=False),
                                     nn.Conv2d(num_channels,num_channels,kernel_size=1),
                                     nn.BatchNorm2d(num_features=num_channels, momentum=0.01, eps=1e-3))

        # Feature scaling layers
        self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.p8_downsample = nn.MaxPool2d(kernel_size=2, stride = 2)

        
        self.first_time = first_time
        if self.first_time:
            self.p8_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[5], num_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p7_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[4], num_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p6_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[3], num_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[2], num_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[1], num_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[0], num_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p7_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w1_relu = nn.ReLU()
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()
        self.p8_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p8_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        outs = self._forward_fast_attention(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5, p6, p7, p8 = inputs
            
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
            p6_in = self.p6_down_channel(p6)
            p7_in = self.p7_down_channel(p7)
            p8_in = self.p8_down_channel(p8)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs

        # P8_0 to P8_2
        
        # Weights for P7_0 and P8_0 to P7_1
        p7_w1 = self.p7_w1_relu(self.p7_w1)
        weight = p7_w1 / (torch.sum(p7_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p7_up = self.conv7_up(swish(weight[0] * p7_in + weight[1] * self.p7_upsample(p8_in)))

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0, P7_1 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0, P7_1 and P6_2 to P7_2
        p7_out = self.conv7_down(
            swish(weight[0] * p7_in + weight[1] * p7_up + weight[2] * self.p7_downsample(p6_out)))
        
        # Weights for P8_0 and P7_2 to P8_2
        p8_w2 = self.p8_w2_relu(self.p8_w2)
        weight = p8_w2 / (torch.sum(p8_w2, dim=0) + self.epsilon)
        # Connections for P8_0 and P7_2 to P8_2
        p8_out = self.conv8_down(swish(weight[0] * p8_in + weight[1] * self.p8_downsample(p7_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out

