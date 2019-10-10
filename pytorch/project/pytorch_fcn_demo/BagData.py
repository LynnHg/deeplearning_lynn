import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
import common

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf


class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('datas/bag_data'))

    def __getitem__(self, idx):
        # 加载训练集
        img_name = os.listdir('datas/bag_data')[idx]
        imgA = cv2.imread('datas/bag_data/'+img_name)
        imgA = cv2.resize(imgA, (common.input_size, common.input_size))
        # 加载mask，即ground truth
        imgB = cv2.imread('datas/bag_data_mask/'+img_name, 0)
        imgB = cv2.resize(imgB, (common.input_size, common.input_size))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)
        if self.transform:
            imgA = self.transform(imgA)    

        return imgA, imgB

bag = BagDataset(transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=common.batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=common.batch_size, shuffle=True, num_workers=4)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch[1])
    for test_batch in test_dataloader:
        print(test_batch)
