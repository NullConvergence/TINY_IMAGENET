import os
import torch
from torchvision import datasets, transforms

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def get_datasets(data_dir, batch_size, apply_transform=True):
    t_trans, tst_trans = get_transforms() if apply_transform is True \
        else get_tensor_transforms()
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_dir, 'train'), t_trans),
        batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_dir, 'val'), t_trans),
        batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(data_dir, 'test'), tst_trans),
        batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    return train_transforms, test_transforms


def get_tensor_transforms():
    print('[INFO][DATA] Getting data without transforms')
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return train_transforms, train_transforms
