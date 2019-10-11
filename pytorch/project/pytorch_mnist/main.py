import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
from models.MyCNN import MyCNN
from models.LeNet5 import LeNet5
import cv2

dst = Dataset()
# 模型实例化
my_model = MyCNN()
my_model = LeNet5()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(my_model.parameters(), lr=1e-4)

# 将模型的所有参数拷贝到到GPU上
if torch.cuda.is_available():
    my_model = my_model.cuda()

def show():
    imgs, labels = next(iter(dst.loader_train))
    # 将一个批次的图拼成雪碧图展示
    # 此时img的维度为[channel, height, width]
    img = torchvision.utils.make_grid(imgs)
    # 转换为numpy数组并调整维度为[height, width, channel]
    # 因为下面的cv2.imshow()方法接受的数据的维度应该这样
    img = img.numpy().transpose(1, 2, 0)
    # 因为之前预处理对数据做了标准差处理
    # 这里需要逆过程来恢复
    img = img * 0.308 + 0.131
    # 打印图片对应标签
    print(labels)
    # 展示图片
    cv2.imshow('mnist', img)
    # 等待图片关闭
    key_pressed = cv2.waitKey(0)

# 为了节省时间成本，这里我们只训练5个epoch
# 可以根据实际情况进行调整
def train(epoches=5):
    for epoch in range(1, epoches + 1):
        print('Epoch {}/{}'.format(epoch, epoches))
        print('-' * 20)

        # 损失值
        running_loss = 0.0
        # 预测的正确数
        running_correct = 0

        for batch, (imgs, labels) in enumerate(dst.loader_train, 1):
            if torch.cuda.is_available():
                # 获取输入数据X和标签Y并拷贝到GPU上
                # 注意有许多教程再这里使用Variable类来包裹数据以达到自动求梯度的目的，如下
                # Variable(imgs)
                # 但是再pytorch4.0之后已经不推荐使用Variable类，Variable和tensor融合到了一起
                # 因此我们这里不需要用Variable
                # 若我们的某个tensor变量需要求梯度，可以用将其属性requires_grad=True,默认值为False
                # 如，若X和y需要求梯度可设置X.requires_grad=True，y.requires_grad=True
                # 但这里我们的X和y不需要进行更新，因此也不用求梯度
                X, y = imgs.cuda(), labels.cuda()
            else:
                X, y = imgs, labels

            # 将输入X送入模型进行训练
            outputs = my_model(X)
            # torch.max()返回两个字，其一是最大值，其二是最大值对应的索引值
            # 这里我们用y_pred接收索引值
            _, y_pred = torch.max(outputs.detach(), dim=1)
            # 在求梯度前将之前累计的梯度清零，以免影响结果
            optimizer.zero_grad()
            # 计算损失值
            loss = loss_fn(outputs, y)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 计算一个批次的损失值和
            running_loss += loss.detach().item()
            # 计算一个批次的预测正确数
            running_correct += torch.sum(y_pred == y)

            # 打印训练结果
            if batch == len(dst.loader_train):
                print(
                    'Batch {batch}/{iter_times},Train Loss:{loss:.2f},Train Acc:{correct}/{lens}={acc:.2f}%'.format(
                        batch=batch,
                        iter_times=len(dst.loader_train),
                        loss=running_loss / batch,
                        correct=running_correct.item(),
                        lens=32 * batch,
                        acc=100 * running_correct.item() / (dst.batch_size * batch)
                    ))
                print('-' * 20)

        if epoch == epoches:
            torch.save(my_model, 'models/MyModels.pth')
            print('Saving models/MyModels.pth')

def test():
    # 加载训练好的模型
    model = torch.load('models/MyModels.pth')
    testing_correct = 0

    for batch, (imgs, labels) in enumerate(dst.loader_test, 1):
        if torch.cuda.is_available():
            X, y = imgs.cuda(), labels.cuda()
        else:
            X, y = imgs, labels
        outputs = model(X)
        _, pred = torch.max(outputs.detach(), dim=1)
        testing_correct += torch.sum(pred == y)
        if batch == len(dst.loader_test):
            print('Batch {}/{}, Test Acc:{}/{}={:.2f}%'.format(
                batch, len(dst.loader_test), testing_correct.item(),
                batch * dst.batch_size, 100 * testing_correct.item() / (batch * dst.batch_size)
            ))

if __name__ == '__main__':
    # train()
    test()