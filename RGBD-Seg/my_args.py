"""
    Arguments available to train scripts.
"""


import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8).')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1).', dest='lr')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam', 'SGD_WARM'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--summary', action='store_true',
                        help="Prints a summary of the model to be trained.")

    # training related
    parser.add_argument('--experiment', required=True,
                        type=str, help="Name of the experiment.")
    parser.add_argument('--debug', action='store_true',
                        help="If used, forces to overfit only one batch of the train split (to debug the network).")

    parser.add_argument('--warmup_steps', dest='warmup_steps', type=int, default=1000)

    # dataset related
    parser.add_argument('--dataset', default='nyuv2',
                        choices=['sunrgbd',
                                 'nyuv2',
                                 'cityscapes', 'cityscapes-with-depth',
                                 'scenenetrgbd'])
    parser.add_argument('--dataset_dir',
                        default=None,
                        help='Path to dataset root.',)
    parser.add_argument('--modality', type=str, default='rgbd', choices=['rgbd', 'rgb', 'depth'])
    parser.add_argument('--raw_depth', action='store_true', default=False,
                        help='Whether to use the raw depth values instead of'
                        'the refined depth values')
    parser.add_argument('--aug_scale_min', default=1.0, type=float,
                        help='the minimum scale for random rescaling the '
                        'training data.')
    parser.add_argument('--aug_scale_max', default=1.4, type=float,
                        help='the maximum scale for random rescaling the '
                        'training data.')
    parser.add_argument('--height', type=int, default=480,
                        help='height of the training images. '
                        'Images will be resized to this height.')
    parser.add_argument('--width', type=int, default=640,
                        help='width of the training images. '
                        'Images will be resized to this width.')
    parser.add_argument('--class_weighting', type=str,
                        default='median_frequency',
                        choices=['median_frequency', 'logarithmic', 'None'],
                        help='which weighting mode to use for weighting the '
                        'classes of the unbalanced dataset'
                        'for the loss function during training.')
    parser.add_argument('--ignore-index', default=0, type=int, help='index to ignore during the evaluation (mIoU) of the experiment')

    return parser.parse_args()
