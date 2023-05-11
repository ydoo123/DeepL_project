from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

custom_transform = transforms.Compose([
    transforms.ToTensor()
])

def make_data_loader(args):
    
    # Get Dataset
    train_dataset = datasets.CIFAR10(args.data, train=True, transform=custom_transform, download=True)
    test_dataset = datasets.CIFAR10(args.data, train=False, transform=custom_transform, download=True)

    # Get Dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader