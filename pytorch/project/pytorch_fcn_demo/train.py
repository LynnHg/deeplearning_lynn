from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from BagData import test_dataloader, train_dataloader
from models.FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet


def train(epo_num=50, show_vgg_params=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCN16s(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epoch in range(1, epo_num + 1):

        train_loss = 0
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])

            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            # print(output)
            # print(bag_msk)
            loss = criterion(output, bag_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):
                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, bag_msk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
                bag_msk_np = np.argmin(bag_msk_np, axis=1)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch:{} train loss = {}, epoch test loss = {}, {}'
              .format(epoch, train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))

        if epoch == epo_num:
            torch.save(fcn_model, 'checkpoints/fcn_model_16.pth')
            print('saveing checkpoints/fcn_model_16.pth')

if __name__ == "__main__":
    train(epo_num=50, show_vgg_params=False)
