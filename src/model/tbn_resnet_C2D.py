# codes are modified from https://github.com/Pika7ma/Temporal-Shift-Module/blob/master/resnet.py
"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'ResNet'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class TB_module(nn.Module):
    def __init__(self, inplanes, planes, stride=1, p=20):
        super(TB_module, self).__init__()
        self.stride = stride
        self.p = p
        bottleneck_planes = int(inplanes / 4)
        self.conv1 = nn.Conv3d(inplanes,
                               bottleneck_planes,
                               kernel_size=(3, 1, 1),
                               padding=(1, 0, 0),
                               stride=(1, stride, stride),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(bottleneck_planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(bottleneck_planes,
                               bottleneck_planes * p,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(bottleneck_planes * p)
        self.conv3 = nn.Conv3d(bottleneck_planes,
                               planes,
                               kernel_size=(3, 1, 1),
                               padding=(1, 0, 0),
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes)

    def temporal_shift(self, x):
        size = x.shape
        out = x
        # check time dimension is bigger than 1
        if (size[3] > 1):
            out[:, :, :, :-1, :, :] = out[:, :, :, 1:, :, :]
        return out

    def forward(self, x, duration):
        # N*T, C, H, W => N, T, C, H, W
        size = x.size()
        x = x.view((-1, duration, size[1], size[2], size[3]))
        # N, T, C, H, W => N, C, T, H, W
        x = x.transpose(1, 2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        origin_size = out.shape
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # reshape
        conv_size = out.shape
        reshape_out = out.view(conv_size[0], origin_size[1], self.p,
                               conv_size[2], conv_size[3], conv_size[4])

        # temporal shift
        ts_out = self.temporal_shift(reshape_out)

        # element-wise multiplication
        out = reshape_out * ts_out

        # sum over factors p
        out = out.sum(2)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        # N, C, T, H, W => N, T, C, H, W
        out = out.transpose(1, 2)
        b, t, c, h, w = out.size()
        out = out.contiguous().view((b * t, c, h, w))

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WideTBBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_tbn=False):
        super(WideTBBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.TB_module = None

        if use_tbn:
            self.TB_module = TB_module(inplanes, planes, stride)

    def forward(self, inp):
        x, duration = inp
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.TB_module is not None:
            tb_out = self.TB_module(x, duration)
            out += tb_out

        out += residual
        out = self.relu(out)

        return [out, duration]


class WideTBBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_tbn=False):
        super(WideTBBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.TB_module = None

        if use_tbn:
            self.TB_module = TB_module(inplanes, planes * self.expansion, stride)

    def forward(self, inp):
        x, duration = inp
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.TB_module is not None:
            tb_out = self.TB_module(x, duration)
            out += tb_out

        out += residual
        out = self.relu(out)

        return [out, duration]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 tbn_layers,
                 num_classes=1000,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], tbn_layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], tbn_layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], tbn_layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], tbn_layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, tbn_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_tbn=(0 < tbn_blocks)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_tbn=(i < tbn_blocks)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # N, C, T, H, W => N*T, C, H, W
        x = x.transpose(1, 2)
        b, t, c, h, w = list(x.shape)
        x = x.contiguous().view((b * t, c, h, w))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, _ = self.layer1([x, t])
        x, _ = self.layer2([x, t])
        x, _ = self.layer3([x, t])
        x, _ = self.layer4([x, t])

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logits = torch.mean(x.view([b, t, -1]), dim=1)
        assert logits.shape[1] == self.num_classes

        return logits


def TBN_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(WideTBBlock, [2, 2, 2, 2], [0, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state = model_zoo.load_url(model_urls['resnet18'])
        del pretrained_state['fc.weight']
        del pretrained_state['fc.bias']
        model.load_state_dict(pretrained_state, strict=False)
    return model


def TBN_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(WideTBBlock, [3, 4, 6, 3], [0, 2, 3, 0], **kwargs)
    if pretrained:
        pretrained_state = model_zoo.load_url(model_urls['resnet34'])
        del pretrained_state['fc.weight']
        del pretrained_state['fc.bias']
        model.load_state_dict(pretrained_state, strict=False)
    return model


def TBN_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(WideTBBottleneck, [3, 4, 6, 3], [0, 0, 2, 0], **kwargs)
    if pretrained:
        pretrained_state = model_zoo.load_url(model_urls['resnet50'])
        del pretrained_state['fc.weight']
        del pretrained_state['fc.bias']
        model.load_state_dict(pretrained_state, strict=False)
    return model


def TBN_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(WideTBBottleneck, [3, 4, 23, 3], [0, 0, 0, 0], **kwargs)
    if pretrained:
        pretrained_state = model_zoo.load_url(model_urls['resnet101'])
        del pretrained_state['fc.weight']
        del pretrained_state['fc.bias']
        model.load_state_dict(pretrained_state, strict=False)
    return model


def TBN_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(WideTBBottleneck, [3, 8, 36, 3], [0, 0, 0, 0], **kwargs)
    if pretrained:
        pretrained_state = model_zoo.load_url(model_urls['resnet152'])
        del pretrained_state['fc.weight']
        del pretrained_state['fc.bias']
        model.load_state_dict(pretrained_state, strict=False)
    return model
