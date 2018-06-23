import numpy as np
import cv2
import torch

#  pip install imgaug
import imgaug as ia
from imgaug import augmenters as iaa


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)


class ToTensor(object):
    def __init__(self, max_objects=50, is_debug=False):
        self.max_objects = max_objects
        self.is_debug = is_debug

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        if self.is_debug == False:
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)

        filled_labels = np.zeros((self.max_objects, 5), np.float32)
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(filled_labels)}

class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        image_new = np.pad(image, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = image_new.shape

        # Extract coordinates for unpadded + unscaled image
        x1 = w * (label[:, 1] - label[:, 3]/2)
        y1 = h * (label[:, 2] - label[:, 4]/2)
        x2 = w * (label[:, 1] + label[:, 3]/2)
        y2 = h * (label[:, 2] + label[:, 4]/2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        label[:, 1] = ((x1 + x2) / 2) / padded_w
        label[:, 2] = ((y1 + y2) / 2) / padded_h
        label[:, 3] *= w / padded_w
        label[:, 4] *= h / padded_h

        return {'image': image_new, 'label': label}

class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = tuple(new_size) #  (w, h)
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, self.new_size, interpolation=self.interpolation)
        return {'image': image, 'label': label}

class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                # Add gaussian noise to some images.
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # Add a value of -5 to 5 to each pixel.
                sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                # Change brightness of images (80-120% of original value).
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                # Improve or worsen the contrast of images.
                sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def __call__(self, sample):
        seq_det = self.seq.to_deterministic()
        image, label = sample['image'], sample['label']
        image = seq_det.augment_images([image])[0]
        return {'image': image, 'label': label}
