from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

custom_transform = transforms.Compose([transforms.ToTensor()])


def make_data_loader(args):
    custom_transform = transforms.Compose([transforms.ToTensor()])

    # Get Dataset
    dataset = datasets.CIFAR10(
        args.data, train=True, transform=custom_transform, download=True
    )

    # Split dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Get DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
