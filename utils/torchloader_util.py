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
from PIL import Image

import numpy as np
import jax

import torch
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup

from absl import logging

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

        img_crop, patches = self.transform_crops(img)

        # turn to tensor and normalize
        img_crop = self.transform(img_crop)
        img_patches = []
        for patch in patches:
            img_patches.append(self.transform(patch).unsqueeze(0))
        img_patches = np.concatenate(img_patches, axis=0)  # [196, 3, p, p]
        shape = img_patches.shape
        img_patches = img_patches.reshape((int(shape[0]**.5),) * 2 + shape[-3:])  # [14, 14, 3, p, p]
        img_patches = img_patches.transpose(2, 0, 1, 3, 4)
        img_patches = img_patches.reshape(img_crop.shape)

        samples = np.concatenate([img_crop[np.newaxis, :, :, :], img_patches[np.newaxis, :, :, :]], axis=0)  # [2, 3, 224, 224]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return samples, target


def build_dataset(is_train, data_dir, aug):
    transform_crops = RandomResizedCropWithPatchAug(
        img_size=aug.img_size,
        scale=aug.area_range,
        ratio=aug.aspect_ratio_range,
        patch_aug=aug.patch_aug,
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
    raise NotImplementedError
    input_size = IMAGE_SIZE

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
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
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


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
class RandomResizedCropWithPatchAug(torch.nn.Module):
    def __init__(self, img_size, patch_aug, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__()
        self.img_size = img_size

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio
        self.patch_aug = patch_aug

        assert img_size % patch_aug.patch_size == 0

    @staticmethod
    def get_params(
            width, height, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        # width, height = F._get_image_size(img)
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
        # step 1: crop the image
        width, height = F._get_image_size(img)
        i, j, h, w = self.get_params(width, height, self.scale, self.ratio)  # i, j: top-left corner
        
        img_crop = F.resized_crop(img, i, j, h, w, (self.img_size,)*2, self.interpolation)

        # step 2: crop the patches from the window of i, j, h, w
        p = self.patch_aug.patch_size
        m = self.img_size // p  # num of patches

        # start points of the patch grid
        xs = ys = np.arange(m) * p
        patches = []
        for y in ys:
            for x in xs:
                patch = F.resized_crop(img_crop, y, x, p, p, p, self.interpolation)
                patches.append(patch)
        return img_crop, patches

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(img_size={0}'.format(self.img_size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', patch_aug={0}'.format(self.patch_aug).replace('\n', '\n' + ' ' * 8)
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