import numpy as np
import scipy.io as sio
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
from main_train import image_h, image_w
import glob
import pdb
import tqdm
import cv2

DATA_ROOT = '/home/xshadow/EarthNets/Dataset4EO/RSUSS'

img_dir_train_dir = os.path.join(DATA_ROOT, 'images', 'train')
depth_dir_train_dir = os.path.join(DATA_ROOT, 'heights', 'train')
label_dir_train_dir = os.path.join(DATA_ROOT, 'classes', 'train')

img_dir_test_dir = os.path.join(DATA_ROOT, 'images', 'test')
depth_dir_test_dir = os.path.join(DATA_ROOT, 'heights', 'test')
label_dir_test_dir = os.path.join(DATA_ROOT, 'classes', 'test')


class NYUv2(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):

        self.phase_train = phase_train
        self.transform = transform

        self.img_dir_train = sorted(glob.glob(img_dir_train_dir+'/*.jpg'))
        self.depth_dir_train = sorted(glob.glob(depth_dir_train_dir+'/*.png'))
        self.label_dir_train = sorted(glob.glob(label_dir_train_dir+'/*.png'))

        self.img_dir_test = sorted(glob.glob(img_dir_test_dir+'/*.jpg'))
        self.depth_dir_test = sorted(glob.glob(depth_dir_test_dir+'/*.png'))
        self.label_dir_test = sorted(glob.glob(label_dir_test_dir+'/*.png'))

        print('Pre-loading...')
        self.images, self.labels, self.heights = [],[],[]
        if not phase_train:
            self.img_dir_train = self.img_dir_test
            self.depth_dir_train = self.depth_dir_test
            self.label_dir_train = self.label_dir_test

        for item in tqdm.tqdm(self.img_dir_train):
            self.images.append(self.read_im(item))
        for item in tqdm.tqdm(self.depth_dir_train):
            self.heights.append(self.read_im(item,2))
        for item in tqdm.tqdm(self.label_dir_train):
            self.labels.append(self.read_im(item,2))

    def read_image(self, path):
        image = h5py.File(path,'r')
        image = image['image'][...]
        #image.flags.writeable = True
        return image

    def read_im(self, path, flag=None):
        if flag:
            image = cv2.imread(path, flag)
        else:
            image = cv2.imread(path)
        return image

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test

        #label = self.read_im(label_dir[idx],2)
        label = self.labels[idx]
        #depth = self.read_im(depth_dir[idx],2)
        depth = self.heights[idx]
        depth = np.nan_to_num(depth)
        depth = depth.clip(0,200)
        #image = self.read_im(img_dir[idx])
        image = self.images[idx]
        #print(image.shape,depth.shape,label.shape)

        sample = {'image': image, 'depth': depth, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        #print(sample['image'].shape,sample['depth'].shape,sample['label'].shape)
        return sample


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255.
        depth = depth / 200.
        #print(image)
        #image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                         std=[0.229, 0.224, 0.225])(image)
        #depth = torchvision.transforms.Normalize(mean=[19050],
        #                                         std=[9650])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #depth = depth.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'label2': torch.from_numpy(label2).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}
