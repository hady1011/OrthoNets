import torch.nn as nn
import torch
import math
import model.attention as Attention
import model.transforms as Transforms


__all__ = ['OrthoNet', 'orthonet18', 'orthonet34', 'orthonet50', 'orthonet101', 'orthonet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,height, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self._process: nn.Module = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.downsample = downsample
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.planes = planes
        self._excitation = nn.Sequential(
            nn.Linear(in_features=planes, out_features=round(planes / 16), device=self.device, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 16), out_features=planes, device=self.device, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention.Attention()
        self.F_C_A = Transforms.GramSchmidtTransform.build(planes, height)
        
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0),out.size(1)
        excitation = self._excitation(compressed).view(b, c, 1, 1)
        attention = excitation * out 
        attention += residual
        activated = torch.relu(attention)
        return activated


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self._process: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.downsample = downsample
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.planes = planes
        self._excitation = nn.Sequential(
            nn.Linear(in_features=4*planes, out_features=round(planes / 4), device=self.device, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 4), out_features=4*planes, device=self.device, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention.Attention()
        self.F_C_A = Transforms.GramSchmidtTransform.build(4 * planes, height)
   
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0),out.size(1)
        attention = self._excitation(compressed).view(b, c, 1, 1)
        attention = attention * out 
        attention += residual
        activated = torch.relu(attention)
        return activated

class OrthoNet(nn.Module):
#make 1000
    def __init__(self, block, layers, num_classes=1000):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inplanes = 64
        super(OrthoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,64, layers[0])
        self.layer2 = self._make_layer(block, 128,32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256,16, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512,8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.to(self._device)

    def _make_layer(self, block, planes,height, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes,height, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,height))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def orthonet18(n_classes, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes, **kwargs)
    return model


def orthonet34(n_classes, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes, **kwargs)
    return model


def orthonet50(n_classes, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes, **kwargs)
    return model


def orthonet101(n_classes, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(Bottleneck, [3, 4, 23, 3], num_classes=n_classes, **kwargs)
    return model


def orthonet152(n_classes, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        n_classes (int) : number of dataset classes
    """
    model = OrthoNet(Bottleneck, [3, 8, 36, 3], num_classes=n_classes, **kwargs)
    return model
