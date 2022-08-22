import os
from pickletools import uint8
import cv2
import torch
import numpy as np
from Dataset4EO.datasets import rsuss
import torch.utils.data as data
import h5py


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._datapipe = rsuss.RSUSS('/home/xshadow/EarthNets/Dataset4EO/',split=split_name, mode='supervised')
        self._file_names = list(self._datapipe)
        self._transform_gt = setting['transform_gt']
        self._x_single_channel = True
        self.class_names = self._datapipe._categories
        self._file_length = None
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        item_name = self._file_names[index]
        rgb_path, x_path, gt_path  = self._file_names[index]

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path)
        gt = self._open_image(gt_path)
        #if self._transform_gt:
        #    gt = self._gt_transform(gt)
        if self._x_single_channel:
            x = self._open_image(x_path)
            x = np.nan_to_num(x)
            x = cv2.merge([x, x, x])

        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def get_length(self):
        return self.__len__()

    def _open_image(self, path):
        file_path = path
        image     = h5py.File(file_path,'r')
        image = image['image'][...]
        image.flags.writeable = True
        return image

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
