import torch
import torchvision
import os
import pickle
import cv2
import time
import torch.nn.functional as F
from dataset import Dataset
from models.ResNet import resnet18
from models.AlexNet import AlexNet
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from tqdm import tqdm

dst = Dataset()
model = resnet18()
# model = AlexNet()

epoches = 10
gpu_is_available = False

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if torch.cuda.is_available():
    gpu_is_available = True
else:
    gpu_is_available = False

if gpu_is_available:
    model = model.cuda()

# device = torch.device('cuda' if gpu_is_available else 'cpu')
# model.to(device)

def timer(func):
    def wrapper(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time() - begin
        print('本次一共花费时间：{:.2f}秒'.format(end))
    return wrapper

def write_csv(results, index, file_name):
    import csv

    print('writing top{} data...'.format(index))
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img', 'score', 'label'])
        writer.writerows(results)
    print('top {} data writed done...'.format(index))

def show():
    # imgs, labels = data_loader()
    # imgs = imgs[0:dst.batch_size]
    # imgs = dst.transform_test(imgs)
    imgs, labels = iter(dst.data_loader_test).next()
    img = torchvision.utils.make_grid(imgs)
    img = img.numpy().transpose((1, 2, 0))

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    img = img * std + mean
    print([dst.classes[i] for i in labels])
    cv2.imshow('img', img)
    cv2.waitKey(0)

def trans_to_percent(num):
    return str(round(num * 100, 2)) + '%'

@timer
def train():
    model.train(True)
    for epoch in range(1, epoches + 1):
        print('Epoch {}/{}'.format(epoch, epoches))
        print('-' * 20)
        running_loss = 0.0
        running_correct = 0
        for batch, data in enumerate(dst.data_loader_train, 1):
            X, y = data
            if gpu_is_available:
                X, y = X.cuda(), y.cuda()
            outputs = model(X)
            _, y_pred = torch.max(outputs, dim=1)
            optimizer.zero_grad()
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
            running_correct += torch.sum(y_pred == y)
            if batch % 3000 == 0:
                print('Batch {}/{}, Train Loss:{:.2f}, Train Acc:{}/{}={:.2f}%'.format(
                    batch,
                    len(dst.data_loader_train),
                    running_loss/batch,
                    running_correct.item(),
                    dst.batch_size * batch,
                    100*running_correct.item()/(dst.batch_size*batch)
                ))

def show_top1(datas):
    # 随机预览
    # 预览图片张数
    top1_size = 20
    for i in range(top1_size):
        index = math.floor(random.random() * len(datas))
        img = datas[index][0]
        # img.shape = [3, 32, 32] -> [32, 32, 3]
        img = img.numpy().transpose((1, 2, 0))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        img = img * std + mean
        score = datas[index][1]
        label = datas[index][2]
        title = score + ' is ' + label
        # 设置图片大小
        plt.rcParams['figure.figsize'] = (4.0, 4.0)
        # 默认间距
        plt.tight_layout()
        # 行，列，索引
        plt.subplot(4, 5, i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=14, color='blue')
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_top_data(top1_data, top5_data):

    # 预览图张数
    top1_size = 20
    top5_size = 4
    # 初始化存图片像素矩阵的sh
    batch_imgs = [0] * top5_size
    category_names = ['accuracy rate', 'error rate']
    batch_results = [0] * top5_size
    for i in range(top5_size):
        # top5 labels
        results = dict()
        index = math.floor(random.random() * len(top5_data))
        img = top5_data[index][0]
        # img.shape = [3, 32, 32] -> [32, 32, 3]
        img = img.numpy().transpose((1, 2, 0))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        img = img * std + mean
        batch_imgs[i] = img
        top5_score = top5_data[index][1]
        top5_label = top5_data[index][2]
        for index, key in enumerate(top5_label):
            results[key] = [100 * round(top5_score[index], 4), 100 * (1 - round(top5_score[index], 4))]
        batch_results[i] = results

    # 子图创建
    # top1 figure
    fig1 = plt.figure(num='Top-1 error', figsize=(7, 7))
    fig1.subplots_adjust(wspace=1)

    # top5 figure
    fig5 = plt.figure(num='Top-5 error', figsize=(7, 7))
    fig5.subplots_adjust(0.1, 0.11, 0.79, 0.88, 0.03, 1)

    # top1可视化
    for i in range(top1_size):
        index = math.floor(random.random() * len(top1_data))
        img = top1_data[index][0]
        # img.shape = [3, 32, 32] -> [32, 32, 3]
        img = img.numpy().transpose((1, 2, 0))
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        img = img * std + mean
        score = top1_data[index][1]
        label = top1_data[index][2]
        title = score + ' is ' + label

        # 默认间距
        plt.tight_layout()
        # 行，列，索引
        ax1 = fig1.add_subplot(4, 5, i + 1)
        ax1.imshow(img)
        ax1.set_title(title, fontsize=10, color='blue')
        ax1.set_xticks([])
        ax1.set_yticks([])

    # 子图序号
    k = 1
    # top5可视化
    while k < top5_size * 2:
        for j in range(top5_size):
            ax5 = fig5.add_subplot(top5_size, 2, k)
            # 去除图片容器的边框
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['bottom'].set_visible(False)
            ax5.spines['left'].set_visible(False)
            # 显示图片
            ax5.imshow(batch_imgs[j])
            # 去除x，y轴刻度
            ax5.set_xticks([])
            ax5.set_yticks([])
            labels = list(batch_results[j].keys())
            data = np.array(list(batch_results[j].values()))
            data_cum = data.cumsum(axis=1)
            category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.85, 0.15, data.shape[1]))
            # info of accuracy an error
            ax = fig5.add_subplot(top5_size, 2, k + 1)
            fig5.tight_layout()
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(data, axis=1).max())

            k += 2

            for i, (colname, color) in enumerate(zip(category_names, category_colors)):
                widths = data[:, i]
                starts = data_cum[:, i] - widths
                ax.barh(labels, widths, left=starts, height=0.8,
                        label=colname, color=color)
                xcenters = starts + widths / 2

                r, g, b, _ = color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                for y, (x, c) in enumerate(zip(xcenters, widths)):
                    ax.text(x, y, str(int(c)) + '%', ha='center', va='center',
                            color=text_color)
            ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                      loc='lower left', fontsize='small')

    plt.show()

def test():
    correct = 0
    test_top1_results = []
    test_top5_results = []
    for batch, (imgs, labels) in enumerate(tqdm(dst.data_loader_test), 1):
        if gpu_is_available:
            X, y = imgs.cuda(), labels.cuda()
            outputs = model(X)
        # top-1 accuracy
        score, pred = torch.max(F.softmax(outputs.cpu(), 1).detach(), 1)
        correct += torch.sum(pred == y.cpu())
        if batch % 100 == 0:
            print('Batch {}/{}, Test Acc:{}/{}={:.2f}%'.format(
                batch, len(dst.data_loader_test), correct.item(),
                batch * dst.batch_size, 100*correct.item()/(batch*dst.batch_size)
            ))
        batch_results = [(_imgs, trans_to_percent(_score.item()), dst.classes[_pred.item()])
                         for _imgs, _score, _pred in zip(imgs, score, pred)]
        test_top1_results += batch_results
        # top-5 accuracy
        batch_scores = F.softmax(outputs.cpu(), 1)
        top = 5
        top5_batch_results = []
        for i in range(len(batch_scores)):
            top5_scores_index = batch_scores.detach().numpy()[i].argsort()[::-1][0:top]
            top5_scores = [batch_scores[i][j] for j in top5_scores_index]
            img = imgs[i]
            top5_batch_results += [(img,[_score.item() for _score in top5_scores],
                                    [dst.classes[_pred.item()] for _pred in top5_scores_index])]
        test_top5_results += top5_batch_results
    write_csv(test_top5_results, 5, 'test_top5_results.csv')
    write_csv(test_top1_results, 1,  'test_top1_results.csv')
    show_top_data(test_top1_results, test_top5_results)

def data_loader():
    import numpy as np
    train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    data = []
    labels = []
    for file_name in train_list:
        path = os.path.join('C:\mydata\cifar-10-batches-py', file_name)
        with open(path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            if 'data' in entry:
                data.extend(entry['data'])
            if 'labels' in entry:
                labels.extend(entry['labels'])
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    # convert to HWC
    data = data.transpose((0, 2, 3, 1))
    return data, labels

if __name__ == '__main__':
    show()




