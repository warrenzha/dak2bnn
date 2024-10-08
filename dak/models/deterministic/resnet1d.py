import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = [
    'ResNet1d', 'resnet1d20', 'resnet1d32', 'resnet1d44', 'resnet1d56', 'resnet1d110',
]


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.skip_add = nn.quantized.FloatFunctional()
        self.shortcut = nn.Sequential()
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
            nn.BatchNorm1d(self.expansion * planes))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.skip_add.add(out, identity)
        # out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block, num_blocks, num_features=16):
        super(ResNet1d, self).__init__()
        self.in_planes = 16
        self.out_features = num_features

        self.conv1 = nn.Conv1d(1,
                               16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU(inplace=True)
        # self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.linear = nn.Linear(64, num_features)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.linear = nn.Linear(16, num_features)
        

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, dim = x.size()
        x = x.view(batch_size, 1, dim) # Reshape to [batch_size, 1, dim]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = F.avg_pool1d(out, out.size()[2]) # Pool over the dim
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet1d20():
    return ResNet1d(BasicBlock1d, [3, 3, 3])


def resnet1d32():
    return ResNet1d(BasicBlock1d, [5, 5, 5])


def resnet1d44():
    return ResNet1d(BasicBlock1d, [7, 7, 7])


def resnet1d56():
    return ResNet1d(BasicBlock1d, [9, 9, 9])


def resnet1d110():
    return ResNet1d(BasicBlock1d, [18, 18, 18])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(lambda p: p.requires_grad and len(p.data.size()) > 1,
                       net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()