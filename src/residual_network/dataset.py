import torchvision
import torchvision.transforms as transforms
import torch

def get_dataloaders():
    # 訓練用データの変形（水増し）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 32pxでランダム切り抜き
        transforms.RandomHorizontalFlip(), # 左右反転
        transforms.ToTensor(), # Tensor型に変換
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 正規化
    ])

    # テスト用データの変形
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10データの読み込み
    train_set = torchvision.datasets.CIFAR10(root="./../../data/residual_network", train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root="./../../data/residual_network", train=False, download=True, transform=transform_test)

    # DataLoaderの作成（バッチ処理用）
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)
    
    # CIFAR-10のクラスラベル
    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, class_names