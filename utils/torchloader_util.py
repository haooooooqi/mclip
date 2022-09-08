# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

import numpy as np
import jax

import torch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# for visualization
MEAN_RGB = [v * 255 for v in IMAGENET_DEFAULT_MEAN]
STDDEV_RGB = [v * 255 for v in IMAGENET_DEFAULT_STD]


from absl import logging

AUTOAUGS = {'autoaug': 'v0', 'randaugv2': 'rand-m9-mstd0.5-inc1'}


class GeneralImageFolder(datasets.ImageFolder):
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
            for _ in range(self.num_views):
                # extra dimension
                samples.append(self.transform(img).unsqueeze(0))
            sample = torch.cat(samples)
        else:
            sample = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def build_dataset(is_train, data_dir, image_size, num_views, aug):
    # build transform
    transform = build_transform(is_train, image_size, aug)

    root = os.path.join(data_dir, 'train' if is_train else 'val')
    dataset = GeneralImageFolder(num_views, root=root, transform=transform)

    logging.info(dataset)

    return dataset


def build_transform(is_train, input_size, aug):
    # train transform
    if is_train:
        color_jitter = None if aug.color_jit is None else aug.color_jit[0]
        aa = AUTOAUGS[aug.autoaug] if aug.autoaug else None
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            scale=aug.area_range,
            ratio=aug.aspect_ratio_range,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation='bicubic',
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
