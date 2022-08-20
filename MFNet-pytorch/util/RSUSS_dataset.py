# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from Dataset4EO.datasets import rsuss
import h5py
import cv2

from ipdb import set_trace as st


class RSUSS_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640 ,transform=[]):
        super(RSUSS_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        self._datapipe = rsuss.RSUSS('/home/xshadow/EarthNets/Dataset4EO/', split=split, mode='supervised')
        self.names = list(self._datapipe)
        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.names)


    def read_image(self, path):
        file_path = path
        image     = h5py.File(file_path,'r')
        image = image['image'][...]
        image.flags.writeable = True
        return image

    def get_train_item(self, index):
        rgb_path, h_path, label_path  = self.names[index]
        image = self.read_image(rgb_path)
        height = self.read_image(h_path)
        height = height.clip(0,200)
        height = cv2.normalize(height, None, 0, 255, cv2.NORM_MINMAX)
        height = np.expand_dims(height, axis=-1)
        label = self.read_image(label_path)
        image = np.concatenate([image, height], axis=-1)
        image = image.astype(np.uint8)

        for func in self.transform:
            image, label = func(image, label)

        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h),Image.NEAREST), dtype=np.int64)

        return torch.tensor(image), torch.tensor(label), rgb_path

    def get_test_item(self, index):
        rgb_path, h_path, label_path = self.names[index]
        image = self.read_image(rgb_path)
        height = self.read_image(h_path)
        height = height.clip(0,200)
        height = cv2.normalize(height, None, 0, 255, cv2.NORM_MINMAX)
        height = np.expand_dims(height, axis=-1)
        image = np.concatenate([image, height], axis=-1)
        image = image.astype(np.uint8)

        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255

        return torch.tensor(image), rgb_path


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    data_dir = '../../data/MF/'
    MF_dataset()
