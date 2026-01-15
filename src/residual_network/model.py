import torch
import torch.nn as nn

# 残差ブロック（Bottleneck型）の定義クラス
class ResBlock(nn.Module):
    def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1):
        super(ResBlock, self).__init__()

        # 1x1 畳み込み：チャネル数を調整
        self.conv1 = nn.Conv2d(first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(first_conv_out_channels) # バッチ正規化

        # 3x3 畳み込み：特徴抽出（stride=2でダウンサンプリング可能）
        self.conv2 = nn.Conv2d(first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(first_conv_out_channels) # バッチ正規化

        # 1x1 畳み込み：チャネル数を4倍に拡大
        self.conv3 = nn.Conv2d(first_conv_out_channels, first_conv_out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(first_conv_out_channels * 4) # バッチ正規化
        self.relu = nn.ReLU() # 活性化関数

        # スキップ接続時の形状調整用レイヤー（必要な場合のみ）
        self.identity_conv = identity_conv

    def forward(self, x):
        identity = x.clone() # 入力を保存（スキップ接続用）

        x = self.conv1(x) # 1段目
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x) # 2段目
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x) # 3段目
        x = self.bn3(x)

        # 入力と出力のサイズが異なる場合、入力を変換
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)

        x += identity # 残差（スキップ）接続
        x = self.relu(x) # 最終的な活性化

        return x