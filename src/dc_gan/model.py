import torch.nn as nn
import config

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 入力は nz (潜在ベクトル)
            nn.ConvTranspose2d(config.nz, config.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.ngf * 8, config.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(config.ngf * 4, config.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(config.ngf * 2, config.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d(config.ngf, config.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 出力: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 入力: (nc) x 64 x 64
            nn.Conv2d(config.nc, config.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
            nn.Conv2d(config.ndf, config.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(config.ndf * 2, config.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(config.ndf * 4, config.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            nn.Conv2d(config.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)