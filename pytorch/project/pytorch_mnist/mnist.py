import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import OrderedDict
import time


# 下载数据集
# 训练数据集 train=True
data_train = datasets.MNIST('./data/mnist',
                            train=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.131], [0.308])
                                ]),
                            download=True
                            )


# 测试数据集 train=False
data_test = datasets.MNIST('./data/mnist',
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),

                           ]),
                           download=True
                           )

# 加载数据集
loader_train = torch.utils.data.DataLoader(data_train,
                                           batch_size=64,
                                           shuffle=True
                                           )
loader_test = torch.utils.data.DataLoader(data_test,
                                           batch_size=64,
                                           shuffle=True
                                         )

class SNN(nn.Module):
    '''
    定义一个3层全连接神经网络结构
    每一层都使用线性函数
    '''
    def __init__(self, inputs, hidden_1, hidden_2, outputs):
        super(SNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(inputs, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_2, outputs),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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
            ('sig8', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


def main():
    model = LeNet5()

    # 将所有的模型参数移动到GPU上
    if torch.cuda.is_available():
        model = model.cuda()

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    epoches = 5

    for epoch in range(epoches):
        running_loss = 0.0
        running_correct = 0
        testing_correct = 0
        print("Epoch  {}/{}".format(epoch+1, epoches))
        for index,data in enumerate(loader_train):
            # imgs.size() = torch.Size([64, 1, 28, 28])
            imgs, labels = data
            # 展开成28*28=784维的向量
            # imgs.size() = torch.Size([64, 784])
            # imgs = imgs.view(imgs.size(0), -1)

            # GPU加速
            if torch.cuda.is_available():
                imgs, labels = imgs.cuda(), labels.cuda()
            else:
                imgs, labels = Variable(imgs), Variable(labels)

            # 开始训练
            outputs = model(imgs)
            # torch.max() 参数1 表示行 0 表示列
            _, pred = torch.max(outputs.data, 1)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            # pred == labels.data返回由1或0组成的张量
            running_correct += torch.sum(pred == labels.data)

        # 模型评估
        # 将model设置为测试模式
        model.eval()
        for data in loader_test:
            imgs, labels = data
            # imgs = imgs.view(imgs.size(0), -1)

            # GPU加速
            if torch.cuda.is_available():
                imgs, labels = imgs.cuda(), labels.cuda()
            else:
                imgs, labels = Variable(imgs), Variable(labels)

            outputs = model(imgs)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == labels.data)

        # testing_correct or running_correct = tensor(x)
        # 当tensor中只有1个元素使，使用.item()转换成标量
        # tensor(x).item() = x
        print("Loss：{:.4f},训练准确率:{:.2f}%,测试准确率:{:.2f}%"
            .format(running_loss / len(data_train),
                   100 * running_correct.item() / len(data_train),
                   100 * testing_correct.item() / len(data_test)))
        print("-" * 50)

def get_mean_std(data_images):
    """Get mean and std by sample ratio
    """
    import numpy as np
    epoch, mean, std = 0, 0, 0
    data_loader = torch.utils.data.DataLoader(data_train,
                                           batch_size=64,
                                           shuffle=True
                                           )
    for imgs, labels in data_loader:
        # imgs.shape = torch.Size([16, 3, 64, 64])
        epoch += 1
        mean += np.mean(imgs.numpy(), axis=(0, 2, 3))
        std += np.std(imgs.numpy(), axis=(0, 2, 3))
        print('epoch:',epoch)

    mean /= epoch
    std /= epoch
    print(mean)
    print(std)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('总耗时{:.4f}s'.format((end-start)))


