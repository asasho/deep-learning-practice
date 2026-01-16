import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

def weights_init(m):
    """
    ニューラルネットワークの重みを初期化する関数
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def show_results(real_batch, img_list, device):
    """
    学習結果（本物の画像と生成された画像）を並べて表示する関数
    """
    plt.figure(figsize=(15,15))
    
    # 本物の画像を表示
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    real_grid = vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu()
    plt.imshow(np.transpose(real_grid, (1, 2, 0)))

    # 生成された画像（学習の最後）を表示
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    fake_grid = img_list[-1] # img_listの最後の要素（最新の生成画像）
    plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
    
    plt.show()