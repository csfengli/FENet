## PyTorch dependencies
import torch
from torchvision import transforms
from Info import Datasets_Info
from utils import *
import os
from datasets.GTOS_mobile import GTOS_mobile_single_data

def Prepare_DataLoaders(opt, split, input_size=(224,224)):
    
    dataset = opt.dataset
    data_dir = Datasets_Info['data_dirs'][dataset]

    train_data_transforms_list = [transforms.Resize(opt.resize_size),
                                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    test_data_transforms_list = [transforms.Resize(opt.center_size),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    if opt.rotation_need:
        test_data_transforms_list.insert(2, transforms.RandomAffine(opt.degree))


    data_transforms = {'train':transforms.Compose(train_data_transforms_list), 'test':transforms.Compose(test_data_transforms_list)}

    # Create training and test datasets
    if dataset == 'GTOS-mobile':
        # Create training and test datasets
        train_dataset = GTOS_mobile_single_data(data_dir, kind = 'train',
                                           image_size=opt.resize_size,
                                           img_transform=data_transforms['train'])

        test_dataset = GTOS_mobile_single_data(data_dir, kind = 'test',
                                           img_transform=data_transforms['test'])

    image_datasets = {'train': train_dataset, 'test': test_dataset}

    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                       batch_size=eval('opt.{}_BS'.format(x)), 
                                                       shuffle=False if x=='test' else True,
                                                       num_workers=opt.num_workers,
                                                       pin_memory=opt.pin_memory) for x in ['train', 'test']}
    
    return dataloaders_dict


