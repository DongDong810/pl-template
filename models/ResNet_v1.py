import torch
import torch.nn as nn


# ResNet (Number of channels in each intermediate block / Repetition of each block / Expansion factor = 4 / Bottleneck layer status)
model_parameters = {}
model_parameters['resnet18'] = ([64,128,256,512] , [2,2,2,2], 1, False)
model_parameters['resnet34'] = ([64,128,256,512], [3,4,6,3], 1, False)
model_parameters['resnet50'] = ([64,128,256,512], [3,4,6,3], 4, True)
model_parameters['resnet101'] = ([64,128,256,512], [3,4,23,3], 4, True)
model_parameters['resnet152'] = ([64,128,256,512], [3,8,36,3], 4, True)


"""
Bottleneck - (conv1x1 -> BN -> relu) -> (conv3x3 -> BN -> relu) -> (conv1x1 -> BN -> relu) -> relu
skip connection - established from the input of a residual block to the point just prior to the final ReLU activation
Identity vs Projected
1) if feature map size same -> Identity function directly
2) if feature map size different -> Projected mapping using 1x1 conv
"""
class Block(nn.Module):

    def __init__(self, in_channels, mid_channels, expansion, is_Bottleneck, stride) -> None:
        """
        Args:
            in_channels (int) : input channels to the Bottleneck
            mid_channels (int) : number of channels to 3x3 conv (intermediate)
            expansion (int) : factor by which the number of input channels are increased
            stride (int) : stride applied in the 3x3 conv. (2 for first Bottleneck of the block and 1 for remaining)
        """

        super(Block, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.is_Bottleneck = is_Bottleneck
        self.relu = nn.ReLU()

        # if dim(x) == dim(F) => Identity function / else => Projected mapping
        if self.in_channels == self.mid_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels*self.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.mid_channels*self.expansion)
            )

        if self.is_Bottleneck:
            # Bottleneck block
            self.conv1_1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(self.mid_channels)
            self.conv2_3x3 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.mid_channels)
            self.conv3_1x1 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)        
            self.bn3 = nn.BatchNorm2d(self.mid_channels*self.expansion)

        else:
            # Basic block
            self.conv1_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.mid_channels)
            self.conv2_3x3 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.mid_channels)


    def forward(self, x):
        # store input to be added before the final relu (resiudal mapping)
        input = x

        if self.is_Bottleneck:
        # Bottleneck block - (conv1x1 -> BN -> relu) -> (conv3x3 -> BN -> relu) -> (conv1x1 -> BN) -> residual -> relu
            x = self.relu(self.bn1(self.conv1_1x1(x)))
            x = self.relu(self.bn2(self.conv2_3x3(x)))
            x = (self.bn3(self.conv3_1x1(x)))

        else:
        # Basic block - (conv3x3 -> BN -> relu) -> (conv3x3 -> BN) -> residual -> relu
            x = self.relu(self.bn1(self.conv1_3x3(x)))
            x = self.bn2(self.conv2_3x3(x))

        if self.identity:
            x += input

        else:
            x += self.projection(input)

        x = self.relu(x)
        return x


# ResNet
class ResNet(nn.Module):
    
    # default = ResNet50
    def __init__(self, cfg, resnet_variant=([64,128,256,512], [3,4,6,3], 4, True), in_channels=1, num_classes=1000) -> None:
        """
        Args:
            resnet_variant (list) : eg. [[64,128,256,512],[3,4,6,3],4,True] - (Number of channels in each intermediate block / Repetition of each block / Expansion factor = 4 / Bottleneck layer status)
            in_channels (int) : image channels (3 - r, g, b)
            num_classes (int) : # of output classes 
        """
        super(ResNet, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repetition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        # 7x7 conv, 64, stride 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # 3x3 max pool, stride 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv block (2~5 in the figure)
        self.block1 = self._make_block(64, self.channels_list[0], self.repetition_list[0], self.expansion, self.is_Bottleneck, stride=1)
        self.block2 = self._make_block(self.channels_list[0]*self.expansion, self.channels_list[1], self.repetition_list[1], self.expansion, self.is_Bottleneck, stride=2)
        self.block3 = self._make_block(self.channels_list[1]*self.expansion, self.channels_list[2], self.repetition_list[2], self.expansion, self.is_Bottleneck, stride=2)
        self.block4 = self._make_block(self.channels_list[2]*self.expansion, self.channels_list[3], self.repetition_list[3], self.expansion, self.is_Bottleneck, stride=2)

        # average pooling -> 1x1
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # fully connected
        self.fc = nn.Linear(in_features=self.channels_list[3]*self.expansion, out_features=num_classes)


    def _make_block(self, in_channels, mid_channels, repeat, expansion, is_Bottleneck, stride):
        """
        Args:
            in_channels : # channels of the Bottleneck input
            mid_channels : # channels of the 3x3 in the Bottleneck
            repeat : # Bottlenecks in the block
            expansion : factor by which mid_channels are multiplied to create the output channels
            is_Bottleneck : status if Bottleneck in required
            stride : stride to be used in the first Bottleneck conv 3x3
        """
        layers = []

        layers.append(Block(in_channels, mid_channels, expansion, is_Bottleneck, stride=stride))
        for num in range(1, repeat):
            layers.append(Block(mid_channels*expansion, mid_channels, expansion, is_Bottleneck, stride=1))

        return nn.Sequential(*layers)


    def forward(self, x):
        ret_dict = {}

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.average_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        ret_dict['y_hat'] = x

        return ret_dict
