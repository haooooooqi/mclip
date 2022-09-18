# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
from absl import logging
import PIL

import torch
import torchvision as tv

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from utils import transform_util

# for visualization
MEAN_RGB = [v * 255 for v in IMAGENET_DEFAULT_MEAN]
STDDEV_RGB = [v * 255 for v in IMAGENET_DEFAULT_STD]

AUTOAUGS = {'autoaug': 'v0', 'randaugv2': 'rand-m9-mstd0.5-inc1'}


class GeneralImageFolder(tv.datasets.ImageFolder):
    def __init__(self, num_views, **kwargs):
        super(GeneralImageFolder, self).__init__(**kwargs)
        self.num_views = num_views

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        body.append("Transform: ")
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transforms)]
        body += ["Number of views: {}".format(self.num_views)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.num_views > 1:
            samples = []
            for i in range(self.num_views):
                if i % 2 == 0:
                    transform = self.transform
                else:
                    transform = self.target_transform
                # extra dimension
                samples.append(transform(img).unsqueeze(0))
            sample = torch.cat(samples)
        else:
            sample = self.transform(img)

        return sample, target


def build_dataset(is_train, data_dir, image_size, num_views, aug):
    # build transform
    transform = build_transform(is_train, image_size, aug)
    # hack: use target transform for the second augmentation
    target_transform = None
    if type(transform) is tuple:
        transform, target_transform = transform

    root = os.path.join(data_dir, 'train' if is_train else 'val')
    dataset = GeneralImageFolder(num_views,
                                root=root,
                                transform=transform,
                                target_transform=target_transform)

    logging.info(dataset)

    return dataset


def build_transform(is_train, input_size, aug):
    if is_train:
        return build_train_transform(input_size, aug)
    else:
        return build_test_transform(input_size, aug)


def build_train_transform(input_size, aug):
    if aug.train_type == 'timm':
        color_jitter = None if aug.color_jit is None else aug.color_jit[0]
        aa = AUTOAUGS[aug.autoaug] if aug.autoaug else None
        # this should always dispatch to transforms_imagenet_train
        return create_transform(
            input_size=input_size,
            is_training=True,
            scale=(aug.area_min, 1.0),
            ratio=aug.aspect_ratio_range,
            hflip=0.5,
            vflip=0.,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation='bicubic',
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    elif aug.train_type == 'moco-v3':
        random_crop = transform_util.RandomResizedCrop(224, scale=(aug.area_min, 1.), iterations=100)
        color_jitter = tv.transforms.ColorJitter(.4, .4, .2, .1)
        rnd_color_jitter = tv.transforms.RandomApply([color_jitter], p=0.8)
        normalize = tv.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        transform = tv.transforms.Compose([
            random_crop,
            tv.transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            tv.transforms.RandomGrayscale(p=0.2),
            tv.transforms.RandomApply([transform_util.GaussianBlurSimple([.1, 2.])], p=1.0),
            tv.transforms.ToTensor(),
            normalize,
        ])
        transform_prime = tv.transforms.Compose([
            random_crop,
            tv.transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            tv.transforms.RandomGrayscale(p=0.2),
            tv.transforms.RandomApply([transform_util.GaussianBlurSimple([.1, 2.])], p=0.1),
            tv.transforms.RandomApply([transform_util.Solarize()], p=0.2),
            tv.transforms.ToTensor(),
            normalize,
        ])
        return transform, transform_prime
    elif aug.train_type == 'moco-v2':
        random_crop = transform_util.RandomResizedCrop(224, scale=(aug.area_min, 1.), iterations=100)
        color_jitter = tv.transforms.ColorJitter(.4, .4, .4, .1)
        rnd_color_jitter = tv.transforms.RandomApply([color_jitter], p=0.8)
        normalize = tv.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        return tv.transforms.Compose([
            random_crop,
            tv.transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            tv.transforms.RandomGrayscale(p=0.2),
            tv.transforms.RandomApply([transform_util.GaussianBlurSimple([.1, 2.])], p=0.5),
            tv.transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError


def build_test_transform(input_size, aug):
    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        tv.transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
    )
    t.append(tv.transforms.CenterCrop(input_size))
    t.append(tv.transforms.ToTensor())
    t.append(tv.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return tv.transforms.Compose(t)
