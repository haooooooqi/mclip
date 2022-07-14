import os
import json

import torch
from torchvision import datasets


class iNaturalist(datasets.VisionDataset):
    """iNaturalist
    Args:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

        split: 'train' or 'val'
        label_smoothing: label smoothing eps
    """
    def __init__(self, split, label_smoothing, **kwargs):

        super(iNaturalist, self).__init__(**kwargs)

        version = os.path.basename(self.root)
        assert version in ['2017', '2018', '2019']

        json_file = os.path.join(self.root, split + version + '.json')  # '/datasets03/inaturalist/2018/val2018.json'
        with open(json_file, "r") as f:
            data = json.load(f)

        assert len(data['images']) == len(data['annotations'])

        self.data = data
        self.num_classes = len(data['categories'])
        self.label_smoothing = label_smoothing
        self.split = split
        self.len = len(data['images'])

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.data['images'][index]['file_name'])
        image = datasets.folder.default_loader(path)

        assert self.data['images'][index]['id'] == self.data['annotations'][index]['id']
        label = self.data['annotations'][index]['category_id']
        assert label < self.num_classes

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), self.num_classes).float()
        label_one_hot = label_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        return image, label, label_one_hot

    def __len__(self) -> int:
        return self.len

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Number of classes: {}".format(self.num_classes))
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        if self.target_transform is not None:
            body.append("Target transform: {}".format(self.target_transform))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)
