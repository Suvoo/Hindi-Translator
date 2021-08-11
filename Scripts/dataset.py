import os
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

# define training and test data directories
data_dir = 'C:\\Users\\91865\\Desktop\\Deva\\DevanagariHandwrittenCharacterDataset'
train_dir = os.path.join(data_dir, 'Train/')
test_dir = os.path.join(data_dir, 'Test/')

# Transforms
def Transforms():

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=train_transform)

    return train_data, test_data


# Loader
def Loader():
    batch_size = 32
    valid_size = 0.10

    num_train = len(train_data)
    split_point = int(valid_size * num_train)

    indices = list(range(num_train))
    np.random.shuffle(indices)

    valid_indices = indices[:split_point]
    train_indices = indices[split_point:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader