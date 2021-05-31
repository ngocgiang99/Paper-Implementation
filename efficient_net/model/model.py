import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import prepare_device

import torch

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DepthwideConv2D(torch.nn.Module):
    def __init__(self, D_in, D_out, kernel_size=3, stride=2):
        super().__init__()
        self.depth_wise = nn.Conv2d(D_in, D_in, kernel_size, stride, kernel_size//2, groups=D_in)
        # self.point_wise = nn.Conv2d(D_in, D_out, 1)

    def forward(self, x):
        x = self.depth_wise(x)
        # x = self.point_wise(x)
        return x

class BottleneckResidualBlock(nn.Module):
    def __init__(self, D_in, expension_rate, D_out, kernel_size=3, stride=2):
        super().__init__()

        self.use_residual = expension_rate == 1 and D_in == D_out

        hidden_dim = expension_rate * D_in

        self.conv1 = nn.Conv2d(D_in, hidden_dim, 1)
        self.norm1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = DepthwideConv2D(hidden_dim, hidden_dim, kernel_size, stride)
        self.norm2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, D_out, 1)
        self.norm3 = nn.BatchNorm2d(D_out)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.relu6(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = F.relu6(y)

        y = self.conv3(y)
        y = self.norm3(y)
        y = F.relu6(y)

        if self.use_residual:
            return x + y
        else:
            return y

class MBConvBlock(nn.Module):
    def __init__(self, D_in, D_out, t, n, kernel_size=3, stride=2):
        super().__init__()

        self.block = nn.Sequential()
        for i in range(n):
            if i == 0:
                self.block.add_module(
                    f'Bottleneck Residual Block {i}', BottleneckResidualBlock(D_in, t, D_out, kernel_size, stride)
                )
            else:
                self.block.add_module(
                    f'Bottleneck Residual Block {i}', BottleneckResidualBlock(D_out, t, D_out, kernel_size, 1)
                )
    def forward(self, x):
        return self.block(x)


class EfficientNetModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model_configure = [
            [32, 16, 1, 1, 3, 1],
            [16, 24, 6, 2, 3, 2],
            [24, 40, 6, 2, 5, 2],
            [40, 80, 6, 3, 3, 2],
            [80, 112, 6, 3, 5, 1],
            [112, 192, 6, 4, 5, 2],
            [192, 320, 6, 1, 3, 1],
        ]

        self.num_classes = num_classes
        self.model = nn.Sequential()

        self.model.add_module(
            'conv3x3_1', nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, stride = 2),
                nn.BatchNorm2d(32),
                nn.ReLU6()
            )
        )

        for i, (d_in, d_out, t, n, k, s) in enumerate(self.model_configure):
            # print(conf)
            # d_in, d_out, t, n, k, s = conf
            # print(d_in, d_out, t, n, k, s)
            self.model.add_module(
                f'MBConv Block {i}', MBConvBlock(d_in, d_out, t, n, k, s)
            )

        self.model.add_module(
            'conv1x1_1', nn.Sequential(
                nn.Conv2d(320, 1280, 1),
                nn.BatchNorm2d(1280),
                nn.ReLU6(),
                nn.AvgPool2d(7),
                nn.Conv2d(1280, num_classes, 1)
            )
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.num_classes)
        return x


if __name__ == '__main__':
    device, device_ids = prepare_device(1)
    print(device, device_ids)

    x = torch.rand(4, 3, 224, 224)
    x = x.to(device)

    model = EfficientNetModel(10)
    model.to(device)

    x = model(x)
    print(x.shape)

    target = torch.randint(10, (4,)).cuda()
    print(target)

    loss = F.cross_entropy(x, target)
    print(loss)

    


