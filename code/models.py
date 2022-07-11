import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(inch, outch, stride=1):
    return nn.Conv2d(inch, outch, kernel_size=3, stride=stride, padding=1, bias=False)


class Block(nn.Module):
    def __init__(self, inch, outch, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = conv3x3(inch, outch, stride)
        self.bn1 = nn.BatchNorm2d(outch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outch, outch)
        self.bn2 = nn.BatchNorm2d(outch)
        self.CONV = None
        if (stride != 1) or (inch != outch):
            self.CONV = nn.Sequential(
                conv3x3(inch, outch, stride=stride),
                nn.BatchNorm2d(outch))

    def forward(self, x):
        temp = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        if self.CONV:
            temp = self.CONV(x)
        out += temp
        out = self.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.layerS1(block, 16, )
        self.layer2 = self.layerS2(block, 32, 2)
        self.layer3 = self.layerS3(block, 64, 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)

    def layerS1(self, block, out_channels, stride=1):
        layers = []
        layers.append(block(16, 16, stride))
        layers.append(block(16, 16))
        layers.append(block(16, 16))
        return nn.Sequential(*layers)

    def layerS2(self, block, out_channels, stride=1):
        layers = []
        layers.append(block(16, 32, stride))
        layers.append(block(32, 32))
        layers.append(block(32, 32))
        return nn.Sequential(*layers)

    def layerS3(self, block, out_channels, stride=1):
        layers = []
        layers.append(block(32, 64, stride))

        layers.append(block(64, 64))
        layers.append(block(64, 64))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNCifar_2(nn.Module):

    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x