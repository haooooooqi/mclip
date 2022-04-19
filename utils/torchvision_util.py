import tensorflow as tf
from tensorflow.python.ops import gen_image_ops

from absl import logging

import torch
from torchvision import transforms
from PIL import Image
import io
import timm
import numpy as np

from utils.transform_util import normalize_image


IMAGE_SIZE = 224
CROP_PADDING = 32


def get_torchvision_aug(image_size, aug):

  transform_aug = [
    transforms.RandomResizedCrop(image_size, scale=aug.area_range, ratio=aug.aspect_ratio_range, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip()]

  if aug.color_jit is not None:
    transform_aug += [transforms.ColorJitter(*aug.color_jit)]

  aa_params = dict(translate_const=int(image_size * 0.45), img_mean=(124, 116, 104),)
  if aug.autoaug == 'randaugv2':
    auto_augment = 'rand-m9-mstd0.5-inc1'
    transform_aug += [timm.data.auto_augment.rand_augment_transform(auto_augment, aa_params)]
  else:
    raise NotImplementedError

  transform_aug += [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

  transform_aug = transforms.Compose(transform_aug)
  return transform_aug


def get_torchvision_aug_eval(image_size):
  transform_aug = [
    transforms.Resize(image_size + CROP_PADDING, interpolation=Image.BICUBIC),  # 256
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
  transform_aug = transforms.Compose(transform_aug)

  return transform_aug


def preprocess_for_train_torchvision(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE, transform_aug=None):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = Image.open(io.BytesIO(image_bytes.numpy()))
  image = image.convert('RGB')

  image = transform_aug(image)
  image = tf.constant(image.numpy(), dtype=dtype)  # [3, 224, 224]
  image = tf.transpose(image, [1, 2, 0])  # [c, h, w] -> [h, w, c]
  return image


def preprocess_for_eval_torchvision(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE, transform_aug=None):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = Image.open(io.BytesIO(image_bytes.numpy()))
  image = image.convert('RGB')

  image = transform_aug(image)
  image = tf.constant(image.numpy(), dtype=dtype)  # [3, 224, 224]
  image = tf.transpose(image, [1, 2, 0])  # [c, h, w] -> [h, w, c]
  return image


def preprocess_for_eval_torchvision_dbg(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE, transform_aug=None):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = Image.open(io.BytesIO(image_bytes.numpy()))
  image = image.convert('RGB')


  # image_tf = tf.io.decode_jpeg(image_bytes, channels=3, fancy_upscaling=False, dct_method='INTEGER_ACCURATE')

  image = np.array(image)
  image = tf.constant(image, dtype=dtype)

  image_height, image_width = image.shape[:2]
  new_size = image_size + CROP_PADDING
  min_size = tf.math.minimum(image_height, image_width)
  new_height = tf.cast(new_size * image_height / min_size, dtype=tf.int32)
  new_width = tf.cast(new_size * image_width / min_size, dtype=tf.int32)

  image = np.array(image)
  image = gen_image_ops.resize_bicubic([image], [new_height, new_width], align_corners=False)[0]
  image = tf.clip_by_value(image, 0, 255.)
  image = tf.round(image)

  image = normalize_image(image)

  offset_height = ((new_height - image_size) + 1) // 2
  offset_width = ((new_width - image_size) + 1) // 2

  image = image[offset_height:(offset_height + image_size), offset_width:(offset_width + image_size), :]

  # -------
  # im = transforms.Resize(image_size + CROP_PADDING, interpolation=Image.BICUBIC)(image)  # PIL, uint8
  # im = transforms.CenterCrop(image_size)(im)  # PIL, uint8
  # im = transforms.ToTensor()(im)  # numpy, float32, [0, 1]



  # transform_aug = [
  #   transforms.Resize(image_size + CROP_PADDING, interpolation=Image.BICUBIC),  # 256
  #   transforms.CenterCrop(image_size),
  #   transforms.ToTensor(),
  #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
  # transform_aug = transforms.Compose(transform_aug)
  # image = transform_aug(image)

  # image = tf.constant(image.numpy(), dtype=dtype)  # [3, 224, 224]
  # image = tf.transpose(image, [1, 2, 0])  # [c, h, w] -> [h, w, c]
  return image


def get_torchvision_map_fn(decode_example):
  # kaiming: reference: https://github.com/tensorflow/tensorflow/issues/38212
  def py_func(image, label):
    d = decode_example({'image': image, 'label': label})
    return list(d.values())
  def ds_map_fn(x):
    flattened_output = tf.py_function(py_func, [x['image'], x['label']], [tf.float32, tf.int64, tf.float32])
    return {"image": flattened_output[0], "label": flattened_output[1], "label_one_hot": flattened_output[2]}
  return ds_map_fn


def get_torchvision_map_mix_fn(aug, num_classes):
  mix_func = timm.data.mixup.Mixup(
    mixup_alpha=aug.mix.mixup_alpha if aug.mix.mixup else 0.0,
    cutmix_alpha=aug.mix.cutmix_alpha if aug.mix.cutmix else 0.0,
    mode='batch',
    label_smoothing=aug.label_smoothing,
    num_classes=num_classes)

  logging.info('Using timm mixup.')

  def apply_mix_func(batch):
    images = batch['image']
    images = torch.tensor(images.numpy())
    images = images.permute([0, 3, 1, 2])  # [n, h, w, c] -> [n, c, h, w]
    
    labels = batch['label']
    labels = torch.tensor(labels.numpy())

    images, labels_one_hot = mix_func(images, labels)
    images = images.permute([0, 2, 3, 1])  # [n, c, h, w] -> [n, h, w, c]
    images = tf.constant(images.numpy(), dtype=batch['image'].dtype)
    labels_one_hot = tf.constant(labels_one_hot.numpy(), dtype=batch['label_one_hot'].dtype)

    return {'image': images, 'label': batch['label'], 'label_one_hot': labels_one_hot}

  # kaiming: reference: https://github.com/tensorflow/tensorflow/issues/38212
  def py_func(image, label, label_one_hot):
    d = apply_mix_func({'image': image, 'label': label, 'label_one_hot': label_one_hot})
    return list(d.values())
  def ds_map_fn(x):
    flattened_output = tf.py_function(py_func, [x['image'], x['label'], x['label_one_hot']], [tf.float32, tf.int64, tf.float32])
    return {"image": flattened_output[0], "label": flattened_output[1], "label_one_hot": flattened_output[2]}
  return ds_map_fn


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



