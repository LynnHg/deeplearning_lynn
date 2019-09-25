import os
import shutil
from config import Config

def redistributionImgs():
    '''
    redistributing all pictures(in './data/Dataset') to the corresponding directory
    '''

    print('Redistribution start...')
    # 调用配置文件
    conf = Config()
    # 如果图片还未分配则进行分配
    if not os.listdir(os.path.join(conf.data_train_root, 'cat/')):
        # 返回所有图片的名字列表
        file_datas = os.listdir(os.path.join(conf.data_root, 'Dataset/'))
        # 使用filter()从有标签的25000张图片中过滤图片
        # 使用匿名函数lambda处理筛选条件
        # 返回标签为狗的图片
        file_dogs = list(filter(lambda x: x[:3] == 'dog', file_datas))
        # 返回标签为猫的图片
        file_cats = list(filter(lambda x: x[:3] == 'cat', file_datas))
        # 有标签的狗图和猫图的个数
        d_len, c_len = len(file_dogs), len(file_cats)
        # 80%的图片用于训练，20%的图片用于验证
        val_d_len, val_c_len = d_len * 0.8, c_len * 0.8

        # 分配狗图片
        for i in range(d_len):
            pre_path = os.path.join(conf.data_root, 'Dataset', file_dogs[i])
            # 80%的数据作为训练数据集
            if i < val_d_len:
                new_path = os.path.join(conf.data_train_root, 'dog/')
            # 20%的数据作为验证数据集
            else:
                new_path = os.path.join(conf.data_valid_root, 'dog/')
            # 调用shutil.move()移动文件
            shutil.move(pre_path, new_path)

        # 分配猫图片
        for i in range(c_len):
            pre_path = os.path.join(conf.data_root, 'Dataset', file_cats[i])
            # 80%的数据作为训练数据集
            if i < val_c_len:
                new_path = os.path.join(conf.data_train_root, 'cat/')
            # 20%的数据作为验证数据集
            else:
                new_path = os.path.join(conf.data_valid_root, 'cat/')
            # 调用shutil.move()移动文件
            shutil.move(pre_path, new_path)
        print('Redistribution completed!')

if __name__ == '__main__':
    redistributionImgs()