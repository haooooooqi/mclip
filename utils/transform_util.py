import math
import numpy as np
import random

import cv2
from PIL import Image, ImageFilter, ImageOps

import torchvision.transforms.functional as tf


def CHW2HWC(image):
    return image.transpose([1, 2, 0])

def HWC2CHW(image):
    return image.transpose([2, 0, 1])

def color_normalization(image, mean, std):
    """Expects image in CHW format."""
    assert len(mean) == image.shape[0]
    assert len(std) == image.shape[0]
    for i in range(image.shape[0]):
        image[i] = image[i] - mean[i]
        image[i] = image[i] / std[i]
    return image

def horizontal_flip(image, prob, order='CHW'):
    assert order in ['CHW', 'HWC']
    if np.random.uniform() < prob:
        if order == 'CHW':
            image = image[:, :, ::-1]
        else:
            image = image[:, ::-1, :]
    return image

def random_crop(image, size):
    if image.shape[0] == size and image.shape[1] == size:
        return image
    height = image.shape[0]
    width = image.shape[1]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = image[y_offset:y_offset + size, x_offset:x_offset + size, :]
    assert cropped.shape[0] == size, "Image not cropped properly"
    assert cropped.shape[1] == size, "Image not cropped properly"
    return cropped

def random_bbox(lam, H, W):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# We blur the image 50% of the time using a Gaussian kernel. We randomly sample
# sigma in [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
class GaussianBlur(object):
    # apply gaussian blur to the image
    def __init__(self, sigma=[.1, 2.], kernel=.1):
        self.sigma = sigma
        self.kernel = kernel

    def __call__(self, x):
        # needs to do some filtering here
        x_cv2 = cv2.cvtColor(np.asarray(x),cv2.COLOR_RGB2BGR)
        h, w, _ = x_cv2.shape
        # kernel size has to be odd
        kh = int(self.kernel * h) // 2 * 2 + 1
        kw = int(self.kernel * w) // 2 * 2 + 1
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x_cv2 = cv2.GaussianBlur(x_cv2, (kh,kw), sigma)
        return Image.fromarray(cv2.cvtColor(x_cv2,cv2.COLOR_BGR2RGB))


class RandomGaussianBlur(object):
    # apply gaussian blur to the image
    def __init__(self, p=0.5, sigma=[.1, 2.], kernel=.1):
        self.p = p
        self.sigma = sigma
        self.kernel = kernel

    def __call__(self, x):
        # needs to do some filtering here
        if np.random.uniform() >= self.p:
            return x
        x_cv2 = cv2.cvtColor(np.asarray(x),cv2.COLOR_RGB2BGR)
        h, w, _ = x_cv2.shape
        # kernel size has to be odd
        kh = int(self.kernel * h) // 2 * 2 + 1
        kw = int(self.kernel * w) // 2 * 2 + 1
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x_cv2 = cv2.GaussianBlur(x_cv2, (kh,kw), sigma)
        return Image.fromarray(cv2.cvtColor(x_cv2,cv2.COLOR_BGR2RGB))


class RandomGaussianBlurFixedKernel(object):
    # apply gaussian blur to the image
    def __init__(self, p=0.5, sigma=[.1, 2.], kernel=5):
        self.p = p
        self.sigma = sigma
        self.kernel = kernel

    def __call__(self, x):
        # needs to do some filtering here
        if np.random.uniform() >= self.p:
            return x
        x_cv2 = cv2.cvtColor(np.asarray(x),cv2.COLOR_RGB2BGR)
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x_cv2 = cv2.GaussianBlur(x_cv2, (self.kernel, self.kernel), sigma)
        return Image.fromarray(cv2.cvtColor(x_cv2,cv2.COLOR_BGR2RGB))


class GaussianBlurSimple(object):
    """Gaussian blur augmentation from SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def _get_image_size(img):
    if tf._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class RandomResizedCrop(object):
    """
    Expose more options
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), iterations=10, interpolation=Image.BILINEAR):
        self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.iterations = iterations
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio, iterations):
        width, height = _get_image_size(img)
        area = height * width

        for _ in range(iterations):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio, self.iterations)
        return tf.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomResizedCropBYOL(object):
    """
    Random Crop from BYOL
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BICUBIC):
        self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = _get_image_size(img)
        area = height * width

        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = random.randint(0, height - h)
        j = random.randint(0, width - w)

        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return tf.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)
