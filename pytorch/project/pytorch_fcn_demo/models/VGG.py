import os

import torch
import torchvision

def VGG16(pretrained=False):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model = torchvision.models.vgg16(pretrained=pretrained)
    state_dict = torch.load(os.path.join(root_dir, 'vgg16-397923af.pth'))
    model.load_state_dict(state_dict)
    # 去掉全连接层
    del model.classifier
    return model

if __name__ == '__main__':
    pass