"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""
import imageio
import torch
from torch.utils import data
import torchvision.utils as v_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
import PIL
from PIL import Image
import os
import scipy.io as sio
import numpy as np
import torchvision.transforms as transforms 

class HimpDataset(data.Dataset):
    def __init__(self, mode, data_root, transform=None):
        self.data = []# read from file, a list of paths
        if len(self.data) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.data_root = data_root

        self.transform = transform

    def __getitem__(self, index):
        img_file, x, y = self.data[index]
        img_path = os.path.join(self.data_root, img_file)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Data augmentation
        # if self.sliding_crop is not None:
        #     img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
        #     if self.transform is not None:
        #         img_slices = [self.transform(e) for e in img_slices]
        #     if self.target_transform is not None:
        #         mask_slices = [self.target_transform(e) for e in mask_slices]
        #     img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
        #     return img, mask, torch.LongTensor(slices_info)
        return img, (x, y)

    def __len__(self):
        return len(self.data)


class HmipDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test', 'random']

        mean_std = ([128.0, 128.0, 128.0], [1.0, 1.0, 1.0])

        self.input_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])

        if self.config.mode == 'random':
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size,
                                     self.config.img_size)
            train_labels = torch.ones(self.config.batch_size, self.config.img_size, self.config.img_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        elif self.config.mode == 'train':
            train_set = HimptDataset('train', self.config.data_root,
                            transform=self.input_transform)
            valid_set = HimptDataset('val', self.config.data_root,
                            transform=self.input_transform)

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size

        elif self.config.mode == 'test':
            test_set = HimptDataset('test', self.config.data_root,
                           transform=self.input_transform)

            self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
                                          num_workers=self.config.data_loader_workers,
                                          pin_memory=self.config.pin_memory)
            self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass

