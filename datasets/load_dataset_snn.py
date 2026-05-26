import os
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch
import global_v as glv


def load_mnist(data_path, batch_size=None, input_size=None, small=False):
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader


def load_fashionmnist(data_path, batch_size=None, input_size=None, small=False):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    return trainloader, testloader


def load_cifar10(data_path, batch_size=None, input_size=None, small=False):
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    return trainloader, testloader


def load_celebA(data_path, batch_size=None, input_size=None, small=False):
    print("loading CelebA")
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])

    test_transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        transforms.CenterCrop(148),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        SetRange])

    trainset = torchvision.datasets.CelebA(root=data_path,
                                           split='train',
                                           download=True,
                                           transform=transform)
    testset = torchvision.datasets.CelebA(root=data_path,
                                            split='test',
                                            download=True,
                                            transform=test_transform)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size*2,
                                            shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader

import cv2
from glob import glob

def load_ad_dataset(data_path, dataset_name, category, batch_size=None, input_size=None):
    print(f"loading {dataset_name} ({category})")
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        
    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']
        
    import torchvision.transforms as transforms
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda X: 2 * X - 1.) # SetRange
    ])
    
    class AD_Dataset(torch.utils.data.Dataset):
        def __init__(self, root, category, split='train'):
            self.transform = transform
            if split == 'train':
                pattern = os.path.join(root, category, 'train', 'good', '*') if dataset_name == 'mvtec' else os.path.join(root, 'visa_pytorch', category, 'train', 'good', '*')
                self.files = sorted(glob(pattern))
                self.labels = [0] * len(self.files)
            else:
                self.files, self.labels = [], []
                test_root = os.path.join(root, category, 'test') if dataset_name == 'mvtec' else os.path.join(root, 'visa_pytorch', category, 'test')
                
                good_dir = os.path.join(test_root, 'good')
                if os.path.exists(good_dir):
                    for f in sorted(glob(os.path.join(good_dir, '*'))):
                        self.files.append(f)
                        self.labels.append(0)
                        
                bad_dir = os.path.join(test_root, 'bad') if dataset_name == 'visa' else test_root
                if dataset_name == 'mvtec':
                    for subfolder in sorted(os.listdir(test_root)):
                        if subfolder == 'good' or not os.path.isdir(os.path.join(test_root, subfolder)): continue
                        for f in sorted(glob(os.path.join(test_root, subfolder, '*'))):
                            self.files.append(f)
                            self.labels.append(1)
                else:
                    if os.path.exists(bad_dir):
                        for f in sorted(glob(os.path.join(bad_dir, '*'))):
                            self.files.append(f)
                            self.labels.append(1)
                            
        def __len__(self):
            return len(self.files)
            
        def __getitem__(self, idx):
            img = cv2.imread(self.files[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform(img), self.labels[idx]

    trainset = AD_Dataset(data_path, category, split='train')
    testset = AD_Dataset(data_path, category, split='test')
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    return trainloader, testloader

def load_mvtec(data_path, category, batch_size=None, input_size=None, small=False):
    return load_ad_dataset(data_path, 'mvtec', category, batch_size, input_size)

def load_visa(data_path, category, batch_size=None, input_size=None, small=False):
    return load_ad_dataset(data_path, 'visa', category, batch_size, input_size)



