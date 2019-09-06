import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from paths import COCO_ROOT


class MiniImagenetCore(Dataset):
    def __init__(self, root, split_file, skip_cls_loc=-1,
                 skip_cls_size=8, transform=None, memoization=True):
        self.memoization = memoization
        self.root = root
        self.transform = transform
        if isinstance(split_file, list):
            annots = pd.concat(pd.read_csv(s) for s in split_file)
        else:
            annots = pd.read_csv(split_file)
        self.classes = annots['label'].values.tolist()
        self.filenames = annots['filename'].values.tolist()
        self.all_classes = sorted(list(set(self.classes)))
        assert len(self.filenames) == len(self.classes)

        if skip_cls_loc >= 0:
            self.split_classes(skip_cls_loc, skip_cls_size)

        self.cats_to_ids = dict(map(reversed, enumerate(self.all_classes)))
        self.ids_to_cats = dict(enumerate(self.all_classes))
        self.labels = [self.cats_to_ids[c] for c in self.classes]
        self.all_labels = [self.cats_to_ids[c] for c in self.all_classes]
        assert len(self.filenames) == len(self.labels)

        if memoization:
            self.name2cache = {}

        print('Created a dataset of {} categories, with {} images'.format(
            len(self.all_classes), len(self.filenames)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        class_name = self.ids_to_cats[self.labels[idx]]

        if self.memoization:
            try:
                img = self.name2cache[name]
            except KeyError:
                img = self.get_data(class_name, name)
                self.name2cache[name] = img
        else:
            img = self.get_data(class_name, name)

        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
            label = torch.tensor(label)
        return {'names': name, 'images': img, 'labels': label}

    def filter_fn(self, l, ind, step):
        if ind >= 0:
            return l[:ind] + l[(ind + step):]
        else:
            return l

    def split_classes(self, skip_cls_loc, skip_cls_size):
        self.all_classes = self.filter_fn(
                self.all_classes, skip_cls_loc, skip_cls_size)
        new_filenames, new_classes = [], []
        for name, cls in zip(self.filenames, self.classes):
            if cls in self.all_classes:
                new_filenames.append(name)
                new_classes.append(cls)
        self.filenames = new_filenames
        self.classes = new_classes
        assert len(self.filenames) == len(self.classes)

    def get_data(self, class_name, name):
        raise NotImplementedError


class MiniImagenetDataset(MiniImagenetCore):
    def __init__(self, *args, **kwargs):
        super(MiniImagenetDataset, self).__init__(*args, **kwargs)
        self.transform = kwargs['transform']

    def get_data(self, class_name, name):
        return Image.open(os.path.join(
                self.root, 'all', class_name, name)).convert('RGB')


class MiniImagenetFeatureset(MiniImagenetCore):
    def __init__(self, *args, **kwargs):
        super(MiniImagenetFeatureset, self).__init__(*args, **kwargs)
        for i, name in enumerate(self.filenames):
            self.filenames[i] = name[:-4] + '.npy'

    def get_data(self, class_name, name):
            return np.load(os.path.join(
                self.root, 'all', class_name, name))


class CocoDataset(object):
    def __init__(self, split='train', transform=None):
        self.root = os.path.join(COCO_ROOT, 'images/train2017')
        self.filenames = os.listdir(self.root)
        self.transform = transform
        print('COCO %s' % split)
        print('Data size ', len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img = Image.open(os.path.join(self.root, name)).convert('RGB')
        label = 0
        if self.transform is not None:
            img = self.transform(img)
            label = torch.tensor(label)
        return {'names': name, 'images': img, 'labels': label}
