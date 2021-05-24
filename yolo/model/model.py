## Template
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

## My code
import torchvision.models as models
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

class YoloV1Backbone():
    def __init__(self):
        self.arch = nn.Sequential()

        self.arch.add(  nn.Conv2d(3, 64, 7, 2, padding=2),
                        nn.MaxPool2d(2, 2))
        self.arch.add(  nn.Conv2d(64, 192, 3),  
                        nn.MaxPool2d(2, 2))

class YoloV1(BaseModel):
    def __init__(self, num_classes=20, S = 7, B = 2):
        super().__init__()
        self.backbone = models.googlenet(True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])
        
        
        self.dense1 = nn.Linear(1024 * 7 * 7, 4096)
        self.dense2 = nn.Linear(4096, S * S * (B * 5 + num_classes))
        self.hyper_param = {    'S': S,
                                'B': B,
                                'C': num_classes}

    def forward(self, x):

        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.dense1(x)
        x = self.dense2(x)

        S = self.hyper_param['S']
        B = self.hyper_param['B']
        C = self.hyper_param['C']
        x = x.view(S, S, 5*B + C)

        return x


if __name__ == '__main__':
    model = YoloV1()

    tensor = torch.rand(1,3,224,224)
    
    x = model(tensor)
    print(x.shape)



