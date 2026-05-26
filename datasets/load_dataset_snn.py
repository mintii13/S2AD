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
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform(img_size, normalize='esvae'):
    transforms_list = [
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if normalize == 'imagenet':
        transforms_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif normalize == 'esvae':
        transforms_list.append(transforms.Lambda(lambda X: 2 * X - 1.))
    return transforms.Compose(transforms_list)

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, split='train', img_size=256, max_samples=None, normalize='esvae'):
        self.transform = get_transform(img_size, normalize)
        self.img_size = img_size
        
        if split == 'train':
            pattern = os.path.join(root, category, 'train', 'good', '*')
            self.files = sorted(glob(pattern))
            self.labels = [0] * len(self.files)
            self.gt_paths = [None] * len(self.files)
        else:
            self.files, self.labels, self.gt_paths = [], [], []
            test_root = os.path.join(root, category, 'test')
            gt_root = os.path.join(root, category, 'ground_truth')
            
            for subfolder in sorted(os.listdir(test_root)):
                fpath = os.path.join(test_root, subfolder)
                if not os.path.isdir(fpath):
                    continue
                lbl = 0 if subfolder == 'good' else 1
                for f in sorted(glob(os.path.join(fpath, '*'))):
                    self.files.append(f)
                    self.labels.append(lbl)
                    
                    if lbl == 1:
                        fname = os.path.splitext(os.path.basename(f))[0]
                        gt_path = None
                        for ext in ['.png', '.bmp', '.jpg']:
                            p = os.path.join(gt_root, subfolder, fname + '_mask' + ext)
                            if os.path.exists(p):
                                gt_path = p
                                break
                            p = os.path.join(gt_root, subfolder, fname + ext)
                            if os.path.exists(p):
                                gt_path = p
                                break
                        self.gt_paths.append(gt_path)
                    else:
                        self.gt_paths.append(None)
        
        if max_samples and max_samples < len(self.files):
            self.files = self.files[:max_samples]
            self.labels = self.labels[:max_samples]
            self.gt_paths = self.gt_paths[:max_samples]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), self.labels[idx], self.gt_paths[idx] or ''

class VisADataset(torch.utils.data.Dataset):
    def __init__(self, root, category, split='train', img_size=256, max_samples=None, normalize='esvae'):
        self.transform = get_transform(img_size, normalize)
        self.img_size = img_size
        
        base = os.path.join(root, 'visa_pytorch', category)
        
        if split == 'train':
            pattern = os.path.join(base, 'train', 'good', '*')
            self.files = sorted(glob(pattern))
            self.labels = [0] * len(self.files)
            self.gt_paths = [None] * len(self.files)
        else:
            self.files, self.labels, self.gt_paths = [], [], []
            test_root = os.path.join(base, 'test')
            gt_root = os.path.join(base, 'ground_truth')
            
            good_dir = os.path.join(test_root, 'good')
            if os.path.exists(good_dir):
                for f in sorted(glob(os.path.join(good_dir, '*'))):
                    self.files.append(f)
                    self.labels.append(0)
                    self.gt_paths.append(None)
            
            bad_dir = os.path.join(test_root, 'bad')
            if os.path.exists(bad_dir):
                for f in sorted(glob(os.path.join(bad_dir, '*'))):
                    self.files.append(f)
                    self.labels.append(1)
                    fname = os.path.splitext(os.path.basename(f))[0]
                    mask_path = os.path.join(gt_root, 'bad', fname + '_mask.png')
                    if not os.path.exists(mask_path):
                        for ext in ['.png', '.bmp', '.jpg']:
                            alt = os.path.join(gt_root, 'bad', fname + ext)
                            if os.path.exists(alt):
                                mask_path = alt
                                break
                    self.gt_paths.append(mask_path if os.path.exists(mask_path) else None)
        
        if max_samples and max_samples < len(self.files):
            self.files = self.files[:max_samples]
            self.labels = self.labels[:max_samples]
            self.gt_paths = self.gt_paths[:max_samples]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), self.labels[idx], self.gt_paths[idx] or ''

def load_mvtec(data_path, category, batch_size=None, input_size=None, small=False, shuffle_train=True, drop_last_train=True, normalize='esvae'):
    print(f"loading MVTec ({category})")
    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']
        
    trainset = MVTecDataset(data_path, category, split='train', img_size=input_size, normalize=normalize)
    testset = MVTecDataset(data_path, category, split='test', img_size=input_size, normalize=normalize)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=4, pin_memory=True, drop_last=drop_last_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return trainloader, testloader

def load_visa(data_path, category, batch_size=None, input_size=None, small=False, shuffle_train=True, drop_last_train=True, normalize='esvae'):
    print(f"loading VisA ({category})")
    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']
        
    trainset = VisADataset(data_path, category, split='train', img_size=input_size, normalize=normalize)
    testset = VisADataset(data_path, category, split='test', img_size=input_size, normalize=normalize)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=4, pin_memory=True, drop_last=drop_last_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return trainloader, testloader



