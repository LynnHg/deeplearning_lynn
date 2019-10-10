import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt
import math
import random
from BagData import test_dataloader, train_dataloader
from models.FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
import common

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('checkpoints/fcn_model_8.pth')  # 加载模型
model = model.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    size = 1
    i = 0
    fig = plt.figure()
    for _ in range(size):
        # index = math.floor(random.random() * 600)
        indexs = [58]
        img_name = r'datas/bag_data/' + str(indexs[_]) + '.jpg'
        imgA = cv2.imread(img_name)
        img = imgA
        imgA = cv2.resize(imgA, (common.input_size, common.input_size))

        mask_name = r'datas/bag_data_mask/' + str(indexs[_]) + '.jpg'
        mask = cv2.imread(mask_name)
        mask = cv2.resize(mask, (common.input_size, common.input_size))

        imgA = transform(imgA)
        imgA = imgA.to(device)
        imgA = imgA.unsqueeze(0)
        output = model(imgA)
        output = torch.sigmoid(output)

        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        print(output_np.shape)  # (1, 2, 160, 160)
        output_np = np.argmin(output_np, axis=1)
        print(output_np.shape)  # (1,160, 160)

        cols = 3
        ax1 = plt.subplot(size, cols, i + 1)
        ax1.set_title('img')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(img)

        ax2 = plt.subplot(size, cols, i + 2)
        ax2.set_title('FCN16s')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(np.squeeze(output_np[0, ...]), 'gray')

        ax3 = plt.subplot(size, cols, i + 3)
        ax3.set_title('Ground truth')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.imshow(mask)

        i += cols
    plt.show()
