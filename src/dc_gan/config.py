import torch
import pathlib

# ハイパーパラメータ
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 32
ndf = 32
num_epochs = 1
lr = 0.0002
beta1 = 0.5
ngpu = 1

# データ・ラベル設定
current_dir = pathlib.Path(__file__).resolve().parent
dataroot = current_dir.parent.parent / "data/dc_gan/images/"
real_label = 1.0
fake_label = 0.0

# デバイス設定
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")