import torch.nn as nn
from collections import OrderedDict

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 这里演示用OrderedDict定义网络模型结果
        # 注意每一层的命名不要重复，不然重复的会不起作用
        self.conv = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('c3', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(2, 2))
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(16 * 5 * 5, 120)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(120, 84)),
            ('relu7', nn.ReLU()),
            ('f8', nn.Linear(84, 10)),
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x