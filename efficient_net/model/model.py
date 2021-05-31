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
class EfficientNetModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()


if __name__ == '__main__':
    device, device_ids = prepare_device(1)
    print(device, device_ids)

    x = torch.rand(10, 32, 28, 28)
    x = x.to(device)
    print(type(x))

    bottle = BottleneckResidualBlock(32, 1, 16, 3, 1)
    bottle.to(device)
    # print(type(bottle))

    model = MnistModel(10)
    # model.to(device)

    # x = model(x)

    x = bottle(x)
    print(x.shape)

