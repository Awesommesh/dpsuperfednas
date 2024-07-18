import pickle
import argparse
import torchvision.transforms as transforms
import numpy as np
import torch
import sys
import os
print(os.getcwd())
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from fedml_api.data_preprocessing.cifar10.datasets import CIFAR10_truncated
from fedml_api.data_preprocessing.cifar100.datasets import CIFAR100_truncated



def add_args(parser):
    # CINIC10 is already split into train, val and test
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to split",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Dataset split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="np random generator seed",
    )
    return parser


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description="split_train_validation"))
    args = parser.parse_args()
    np.random.seed(args.seed)
    split_ratio = args.split_ratio
    X_train, y_train = None, None
    datadir = "cifar10/"

    if args.dataset == "cifar10":
        train_transform, _ = _data_transforms_cifar10()
        cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
        X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target

    elif args.dataset == "cifar100":
        train_transform, _ = _data_transforms_cifar100()
        datadir = "cifar100/"
        cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=train_transform)

        X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    else:
        raise ValueError("Unexpected dataset: {}".format(args.dataset))

    # Begin splitting train into train and validation
    new_train_len = int(split_ratio * len(X_train))
    random_shuffle = np.random.permutation(len(X_train))
    X_train_post_val = X_train[random_shuffle[:new_train_len]]
    y_train_post_val = y_train[random_shuffle[:new_train_len]]
    X_val = X_train[random_shuffle[new_train_len:]]
    y_val = y_train[random_shuffle[new_train_len:]]
    new_train_set = (X_train_post_val, y_train_post_val)
    gen_val_set = (X_val, y_val)
    with open(datadir + 'train.pkl', 'wb') as file:
        pickle.dump(new_train_set, file)
    with open(datadir + 'val.pkl', 'wb') as file:
        pickle.dump(new_train_set, file)