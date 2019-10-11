import torch
from torchvision import datasets, transforms

class Dataset():
    def __init__(self):
        # 注意这是python 2.0的写法
        super(Dataset, self).__init__()
        # python 3.0+可省略super()中的参数
        #  super().__init__()

        # 一个批次加载的图片数量
        self.batch_size = 64
        # 数据预处理
        # Compose用于将多个transfrom组合起来
        # ToTensor()将像素转换为tensor，并做Min-max归一化，即x'=x-min/max-min
        # 相当于将像素从[0,255]转换为[0,1]
        # Normalize()用均值和标准差对图像标准化处理 x'=(x-mean)/std，加速收敛的作用
        # 这里0.131是图片的均值，0.308是方差，通过对原始图片进行计算得出
        # 想偷懒的话可以直接填Normalize([0.5], [0.5])
        # 另外多说一点，因为MNIST数据集图片是灰度图，只有一个通道，因此这里的均值和方差都只有一个值
        # 若是普通的彩色图像，则应该是三个值，比如Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.131], [0.308])])
        # 下载数据集
        # 训练数据集 train=True
        # './data/mnist'是数据集存放的路径，可自行调整
        # download=True表示叫pytorch帮我们自动下载
        self.data_train = datasets.MNIST('./data/mnist',
                                    train=True,
                                    transform=self.transforms,
                                    download=True
                                    )

        # 测试数据集 train=False
        self.data_test = datasets.MNIST('./data/mnist',
                                   train=False,
                                   transform=self.transforms,
                                   download=True
                                   )

        # 加载数据集
        # shuffle=True表示加载时打乱图片顺序，有一定的防止过拟合效果
        self.loader_train = torch.utils.data.DataLoader(self.data_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        # 测试集就不需要打乱了，因此shuffle=False
        self.loader_test = torch.utils.data.DataLoader(self.data_test,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

if __name__ == '__main__':
    dst = Dataset()