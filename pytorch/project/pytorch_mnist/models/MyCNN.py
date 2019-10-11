import torch.nn as nn

class MyCNN(nn.Module):
    # 因为分10类，设置n_classes=10
    def __init__(self, n_classes=10):
        super(MyCNN, self).__init__()

        # 关于pytorch中网络层次的定义有几种方式，这里用的其中一种，用nn.Sequential()进行组合
        # 另外还可用有序字典OrderedDict进行定义
        # 再或者不使用nn.Sequential()进行组合，而是每一层单独定义
        # 看具体需求和个人爱好
        self.features = nn.Sequential(
            # 输入28×28，灰度图in_channels=1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=0),  # 输出22×22
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出11×11
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 11 * 11, 160),
            nn.Linear(160, 80),
            nn.Linear(80, n_classes)
        )

    # 定义前向传播函数
    def forward(self, x):
        # 将输入送入卷积核池化层
        out = self.features(x)
        # 这里需要将out扁平化，展开成一维向量
        # 具体可惨开view()的用法
        out = out.view(out.size()[0], -1)
        # 将卷积和池化后的结果送入全连接层
        out = self.classifier(out)

        return out

if __name__ == '__main':
    print(MyCNN())