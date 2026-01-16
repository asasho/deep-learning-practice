from torchvision import transforms, datasets
import torch.utils.data
import config

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(root=config.dataroot, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.workers
    )
    return dataloader