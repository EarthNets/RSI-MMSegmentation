# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from ipdb import set_trace as st


class Potsdam_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640 ,transform=[]):
        super(Potsdam_dataset, self).__init__()

        assert split in ['train', 'test'], 'split must be "train"|"test"'
        self.image_root = "/home/xshadow/EarthNets/Dataset4EO/Potsdam/Images/resized/"
        self.label_root = "/home/xshadow/EarthNets/Dataset4EO/Potsdam/Labels/labeled/"
        self.height_root = "/home/xshadow/EarthNets/Dataset4EO/Potsdam/Heights/resized/"
        split_files = {
        "train": [
            "top_potsdam_2_10",
            "top_potsdam_2_11",
            "top_potsdam_2_12",
            "top_potsdam_3_10",
            "top_potsdam_3_11",
            "top_potsdam_3_12",
            "top_potsdam_4_10",
            "top_potsdam_4_11",
            "top_potsdam_4_12",
            "top_potsdam_5_10",
            "top_potsdam_5_11",
            "top_potsdam_5_12",
            "top_potsdam_6_10",
            "top_potsdam_6_11",
            "top_potsdam_6_12",
            "top_potsdam_6_7",
            "top_potsdam_6_8",
            "top_potsdam_6_9",
            "top_potsdam_7_10",
            "top_potsdam_7_11",
            "top_potsdam_7_12",
            "top_potsdam_7_7",
            "top_potsdam_7_8",
            "top_potsdam_7_9",],
        "test": [
            "top_potsdam_5_15",
            "top_potsdam_6_15",
            "top_potsdam_6_13",
            "top_potsdam_3_13",
            "top_potsdam_4_14",
            "top_potsdam_6_14",
            "top_potsdam_5_14",
            "top_potsdam_2_13",
            "top_potsdam_4_15",
            "top_potsdam_2_14",
            "top_potsdam_5_13",
            "top_potsdam_4_13",
            "top_potsdam_3_14",
            "top_potsdam_7_13",],
        }

        self.class_names = [
        "background",
        "Impervious surfaces",
        "Building",
        "Low Vegetation",
        "Tree",
        "Car",
        ]

        split_files = self._extend_tiles(split_files)

        self.data_dir  = data_dir
        self.split     = split
        self.names = split_files[self.split]
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.names)


    def _extend_tiles(self, files):
        train_names = []
        test_names = []
        for split in ["train","test"]:
            for it in files[split]:
                name1 = it+'_RGB_01_01.png'
                name2 = it+'_RGB_01_02.png'
                name3 = it+'_RGB_02_01.png'
                name4 = it+'_RGB_02_02.png'
                if split == "train":
                    train_names.append(name1)
                    train_names.append(name2)
                    train_names.append(name3)
                    train_names.append(name4)
                elif split == "test":
                    test_names.append(name4)
                    test_names.append(name4)
                    test_names.append(name4)
                    test_names.append(name4)
        out_files = {}
        out_files['train'] = train_names
        out_files['test'] = test_names
        return out_files

    def read_image(self, name, M='rgb'):
        if M == 'rgb':
            file_path = self.image_root+name
        elif M == 'h':
            names = name.split("_")
            names[0] = 'dsm'
            names[1] = 'potsdam'
            names[2] = names[2].zfill(2)
            names[3] = names[3].zfill(2)
            name = "_".join(names)
            file_path = self.height_root+name.replace('RGB','normalized_lastools')
        elif M == 'label':
            file_path = self.label_root+name.replace('RGB','label')
        image     = np.asarray(Image.open(file_path)) # (w,h,c)
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, M='rgb')
        height = self.read_image(name, M='h')
        label = self.read_image(name, M='label')
        height = height[:,:,0]
        height = np.expand_dims(height, axis=-1)
        #print(image.shape,height.shape)
        image = np.concatenate([image, height], axis=-1)
        image = image.astype(np.uint8)

        for func in self.transform:
            image, label = func(image, label)

        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h)), dtype=np.int64)
        #print(np.unique(label))

        return torch.tensor(image), torch.tensor(label), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    data_dir = '../../data/MF/'
    Potsdam_dataset()
