import torch
import torchvision
from torchvision import datasets, transforms


class Dataset:
    def __init__(self):
        self.batch_size = 16
        self.transform_train = transforms.Compose([transforms.Resize(32),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.transform_test = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.data_train = datasets.CIFAR10(root='./datas', train=True, download=False, transform=self.transform_train)
        self.data_test = datasets.CIFAR10(root='./datas', train=False, download=False, transform=self.transform_test)

        self.data_loader_train = torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)
        self.data_loader_test = torch.utils.data.DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
