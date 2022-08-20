import numpy as np
from torch import nn
import torch
import os
import torch.nn.functional as F
import pdb
import scipy.ndimage as nd


med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773]


med_frq = [0.272491, 0.568953, 0.432069, 0.354511,
           0.82178, 0.506488, 1.133686, 0.81217, 0.789383,
           0.380358, 1.650497, 1, 0.650831, 0.757218,
           0.950049, 0.614332, 0.483815, 1.842002, 0.635787,
           1.176839, 1.196984, 1.111907, 1.927519, 0.695354,
           1.057833, 4.179196, 1.571971, 0.432408, 3.705966,
           0.549132, 1.282043, 2.329812, 0.992398, 3.114945,
           5.466101, 1.085242, 6.968411, 1.093939, 1.33652,
           1.228912]

# 0.342837

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_colours = [(0, 0, 0),
                 # 0=background
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]

def cross_entropy_loss(input, target, reduction, weight):
    logp = F.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target * weight, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')



class OhemCrossEntropy2d(nn.Module):

    def __init__(self, thresh=0.6, min_kept=200000, factor=8, weight=None):
        super(OhemCrossEntropy2d, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.ignore_label = -1

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0/factor, 1.0/factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor*factor) #int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        label = input_label
        prob = input_prob
        pred = prob[label, np.arange(len(label), dtype=np.int32)]
        threshold = self.thresh
        if min_kept > 0:
            k_th = min(len(pred), min_kept)-1
            new_array = np.partition(pred, k_th)
            new_threshold = new_array[k_th]
            if new_threshold > self.thresh:
                threshold = new_threshold
        return threshold


    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label >= 0
        valid_inds = np.where(valid_flag)[0]
        label = input_label
        #print('All Labels: {}'.format(len(valid_inds)))

        prob = input_prob
        pred = prob[label, np.arange(len(label), dtype=np.int32)]
        kept_flag = pred <= threshold
        valid_inds = valid_inds[kept_flag]
        #print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target


    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return target


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
        #self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction = 'none')
        self.ohem = OhemCrossEntropy2d()

    def forward(self, inputs_scales, targets_scales):
        losses = []
        ct = 0
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone() + 0.5
            targets_m[mask] -= 1.0
            if ct == 0:
                new_targets = self.ohem(inputs, targets_m.long())
                new_mask = new_targets != -1
                new_mask = mask & new_mask
                loss_all = self.ce_loss(inputs, targets_m.long())
                losses.append(torch.sum(torch.masked_select(loss_all, new_mask)) / torch.sum(new_mask.float()))
            else:
                loss_all = self.ce_loss(inputs, targets_m.long())
                losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
            ct += 1
        total_loss = sum(losses)
        return total_loss


class FocalLoss2d(nn.Module):
    def __init__(self, weight=med_frq, gamma=2.0):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = torch.from_numpy(np.asarray(weight)).float()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction = 'none')
        self.fl_loss = FocalLoss(gamma=self.gamma, alpha=self.weight)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone() + 0.5
            targets_m[mask] -= 1.0
            loss_all = 0.5 * self.fl_loss(inputs, targets_m.long()) + 0.5 * self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)

# added by hxx for iou calculation
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1 # hxx
    # imLab += 1 # label
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def accuracy(preds, label):
    valid = (label > 0) # hxx
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def macc(preds, label, num_class):
    a = np.zeros(num_class)
    b = np.zeros(num_class)
    for i in range(num_class):
        mask = (label == i+1)
        a_sum = (mask * preds == i+1).sum()
        b_sum = mask.sum()
        a[i] = a_sum
        b[i] = b_sum
    return a,b

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
