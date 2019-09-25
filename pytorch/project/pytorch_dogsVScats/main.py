import torch
import time
import cv2
import os
import random
import math
import torchvision
import torch.nn as nn
from data.dataset import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from models.VGG import VGG16
from models.GoogLeNet import GoogLeNet
from models.AlexNet import AlexNet
from config import Config
import matplotlib.pyplot as plt


# 数据类实例
dst = Dataset()
# 配置类实例
conf = Config()
# 模型类实例
model = AlexNet()

# 定义超级参数
epoch_n = 1

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 如果GPUs可用，则将模型上需要计算的所有参数复制到GPUs上
if torch.cuda.is_available():
    model = model.cuda()

# 代码执行时间装饰器
def timer(func):
    def wrapper(*args, **kw):
        begin = time.time()
        # 执行函数体
        func(*args, **kw)
        end = time.time()

        # 花费时间
        cost = end - begin
        print('本次一共花费时间：{:.2f}秒。'.format(cost))
    return wrapper

# 保存测试结果
def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label', 'pred'])
        writer.writerows(results)

# 图片预览
def show():
    # 图片预览
    imgs, labels = iter(dst.data_images_loader['train']).next()
    # 制作雪碧图
    # 类型为tensor，维度为[channel, height, width]
    img = torchvision.utils.make_grid(imgs)
    # 转换为数组并调整维度为[height, width, channel]
    img = img.numpy().transpose((1, 2, 0))
    # 通过反向推导标准差交换法计算图片原来的像素值
    mean, std = conf.mean, conf.std
    img = img * std + mean
    # 打印图片标签
    print([dst.classes[i] for i in labels])
    # 显示图片
    cv2.imshow('img', img)
    # 等待图片关闭
    cv2.waitKey(0)

# 训练
@timer
def train():
    for epoch in range(1, epoch_n + 1):
        print('Epoch {}/{}'.format(epoch, epoch_n))
        print('-'*20)

        for phase in ['train', 'valid']:
            if phase == 'train':
                print('Training...')
                # 打开训练模式
                model.train(True)
            else:
                print('Validing...')
                # 关闭训练模式
                model.train(False)

            # 损失值
            running_loss = 0.0
            # 预测的正确数
            running_correct = 0
            # 让batch的值从1开始，便于后面计算
            for batch, data in enumerate(dst.data_images_loader[phase], 1):
                # 实际输入值和输出值
                X, y = data
                # 将参数复制到GPUs上进行运算
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                # outputs.shape = [32,2] -> [1,2]
                outputs = model(X)
                # 从输出结果中取出需要的预测值
                _, y_pred = torch.max(outputs.detach(),  1)
                # 将Varibale的梯度置零
                optimizer.zero_grad()
                # 计算损失值
                loss = loss_fn(outputs, y)
                if phase == 'train':
                    # 反向传播求导
                    loss.backward()
                    # 更新所有参数
                    optimizer.step()

                running_loss += loss.detach().item()
                running_correct += torch.sum(y_pred == y)
                if batch % 500 == 0 and phase == 'train':
                    print('Batch {}/{},Train Loss:{:.2f},Train Acc:{:.2f}%'.format(
                        batch, len(dst.data_images[phase])/conf.batch_size, running_loss/batch, 100*running_correct.item()/(conf.batch_size*batch)
                    ))
            epoch_loss = running_loss*conf.batch_size/len(dst.data_images[phase])
            epoch_acc = 100*running_correct.item()/len(dst.data_images[phase])
            print('{} Loss:{:.2f} Acc:{:.2f}%'.format(phase, epoch_loss, epoch_acc))

def show_test_result(results):
    # 随机预览
    # 预览图片张数
    size = 20
    for i in range(size):
        index = math.floor(random.random() * len(results))
        # 图片名称
        file_name = str(results[index][0]) + '.jpg'
        # 图片是否个对象的概率
        score = results[index][1]
        # 'cat' or 'dog'
        name = results[index][2]
        # 获取图片
        img = cv2.imread(os.path.join(conf.data_test_root, file_name))
        title = score + ' is ' + name

        # 设置图片大小
        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        # 默认间距
        plt.tight_layout()
        # 行，列，索引
        plt.subplot(4, 5, i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=14, color='blue')
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 测试
def test():
    dst = Dataset(train=False)
    data_loader_test = torch.utils.data.DataLoader(dst,
                                                   batch_size=conf.batch_size,
                                                   shuffle=False)
    # 保存测试结果
    results = []
    # tqdm模块用于显示进度条
    for imgs, path in tqdm(data_loader_test):
        if torch.cuda.is_available():
            X = imgs.cuda()
        outputs = model(X)
        # pred表示是哪个对象，0=cat，1=dog
        # probability表示是否个对象的概率
        probability, pred = torch.max(F.softmax(outputs, dim=1).detach(), dim=1)
        # 通过zip()打包为元组的列表,如[(1001, 23%, 'cat')]
        batch_results = [(path_.item(), str(round(probability_.item()*100, 2))+'%', 'dog'if pred_.item() else 'cat')
                         for path_, probability_, pred_ in zip(path, probability, pred)]
        results += batch_results
    write_csv(results, conf.result_file)

    show_test_result(results)


if __name__ == '__main__':
    # train
    test()