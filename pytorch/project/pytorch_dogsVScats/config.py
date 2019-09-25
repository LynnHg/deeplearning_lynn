class Config:
    # 文件路径
    # 数据集根目录
    data_root = './datas/'
    # 训练集存放路径
    data_train_root = './datas/train/'
    # 验证集存放路径
    data_valid_root = './datas/valid/'
    # 测试集存放路径
    data_test_root = './datas/test/'

    # 测试结果保存位置
    result_file = './result.csv'

    # 常用参数
    # 图片大小
    input_size = 227
    # batch size
    batch_size = 1
    # mean and std
    # 通过抽样计算得到图片的均值mean和标准差std
    mean = [0.470, 0.431, 0.393]
    std = [0.274, 0.263, 0.260]

    # 预训练模型路径
    path_vgg16 = './models/vgg16-397923af.pth'