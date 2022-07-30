# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from re import T
from absl import logging
import os
import PIL
from PIL import Image

import numpy as np
import jax

import torch
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from torch import Tensor

import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup

import math
import numbers
import warnings
from collections.abc import Sequence
from typing import Tuple, List

IMAGE_SIZE = 224

AUTOAUGS = {'autoaug': 'v0', 'randaugv2': 'rand-m9-mstd0.5-inc1'}


class GeneralImageFolder(datasets.ImageFolder):
    def __init__(self, transform_crops, **kwargs):
        super(GeneralImageFolder, self).__init__(**kwargs)
        self.transform_crops = transform_crops

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()

        body.append("** transform_crops **:")
        if hasattr(self, "transform_crops") and self.transform_crops is not None:
            body += [repr(self.transform_crops)]

        body.append("** transform **:")
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path)

        patches = self.transform_crops(img)


        sample = self.transform(img).unsqueeze(0)
        sample_view0 = self.second_transform(img).unsqueeze(0)
        sample_view1 = self.second_transform(img).unsqueeze(0)

        sample = torch.cat([sample, sample_view0, sample_view1])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def build_dataset(is_train, data_dir, aug):

    transform_crops = RandomPatchCrop(
        patch_size=aug.patch_size,
        super_size=aug.super_size,
        ref_img_size=224,
        scale=aug.area_range, ratio=aug.aspect_ratio_range,
        interpolation=Image.BICUBIC)

    # other non-geometric transforms
    t = []
    if aug.color_jit:
        t.append(transforms.ColorJitter(*aug.color_jit))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(t)
    
    root = os.path.join(data_dir, 'train' if is_train else 'val')
    dataset = GeneralImageFolder(root=root, transform=transform, transform_crops=transform_crops)

    logging.info(dataset)

    return dataset


def build_transform(is_train, aug):
    input_size = IMAGE_SIZE

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train:  # train transform
        color_jitter = None if aug.color_jit is None else aug.color_jit[0]
        aa = AUTOAUGS[aug.autoaug] if aug.autoaug else None
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=IMAGE_SIZE,
            is_training=True,
            scale=aug.area_range,
            ratio=aug.aspect_ratio_range,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation='bicubic',
            mean=mean,
            std=std,
        )
        from IPython import embed; embed();
        if (0 == 0): raise NotImplementedError
        return transform
    else:  # eval transform
        raise NotImplementedError
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
        t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_mixup_fn(aug, num_classes=1000):
    mixup_fn = None
    mixup_active = aug.mix.mixup or aug.mix.cutmix
    if mixup_active:
        logging.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=aug.mix.mixup_alpha,
            cutmix_alpha=aug.mix.cutmix_alpha,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=aug.label_smoothing,
            num_classes=num_classes)
    return mixup_fn


def prepare_pt_data(xs, batch_size):
  """Convert a input batch from PyTorch Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x.numpy()  # pylint: disable=protected-access

    if x.shape[0] != batch_size:
      pads = -np.ones((batch_size - x.shape[0],) + x.shape[1:], dtype=x.dtype)
      x = np.concatenate([x, pads], axis=0)

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


# overwrite timm
def one_hot(x, num_classes, on_value=1., off_value=0., device='cpu'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cpu'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)

timm.data.mixup.one_hot = one_hot
timm.data.mixup.mixup_target = mixup_target


# ------------------------------------------------------------
# Random Patch Crop Impl
# ------------------------------------------------------------
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class RandomPatchCrop(torch.nn.Module):
    def __init__(self, patch_size, super_size, ref_img_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__()
        self.patch_size = _setup_size(patch_size, error_msg="")
        self.super_size = _setup_size(super_size, error_msg="")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation

        # As if to take a 224 crop first, and then take a super_size*super_size from this crop
        area_rescale = (super_size / ref_img_size) ** 2
        self.scale = (scale[0] * area_rescale, scale[1] * area_rescale)
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(patch_size={0}'.format(self.patch_size)
        format_string += self.__class__.__name__ + '(super_size={0}'.format(self.super_size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}