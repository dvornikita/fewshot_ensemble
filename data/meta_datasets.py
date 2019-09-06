import os
import numpy as np
from glob import glob
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from utils import read_textfile
from utils import merge_dicts
import random


class OmniglotDataset(Dataset):
    def __init__(self, root, split_file, n_train, n_test, transform=None):
        self.root = root
        self.all_classes = read_textfile(split_file)
        self.n_train, self.n_test = n_train, n_test
        self.transform = transform

        base_classes = set('/'.join(s.split('/')[:-1])
                           for s in self.all_classes)
        self.class2data = {c: glob(os.path.join(root, c, '*.png'))
                           for c in base_classes}
        self.name2cache = {}

    def __len__(self):
        return len(self.all_classes)

    def __getitem__(self, idx):
        rot_class = self.all_classes[idx]
        cls_split = rot_class.split('/')
        base_class = '/'.join(cls_split[:2])
        angle = float(cls_split[2][3:])
        img_list = self.class2data[base_class]

        names = random.sample(img_list, self.n_train + self.n_test)
        imgs, labels = [], []
        for name in names:
            try:
                img = self.name2cache[name]
            except KeyError:
                img = Image.open(name)
                self.name2cache[name] = img
            img = self.oblig_transform(img, angle)
            img = img if self.transform is None else self.transform(img)
            imgs.append(img)
            labels.append(idx)
        return {'images': imgs, 'labels': labels}

    def oblig_transform(self, img, angle):
        img = img.resize((28, 28))
        img = img.rotate(angle)
        return img


class MiniImagenetClass(Dataset):
    def __init__(self, names_list, class_name, root, data_type,
                 transform=None, test_transform=None):

        self.root = root
        self.all_names = names_list
        self.class_name = class_name
        self.data_type = data_type
        self.name2cache = dict()
        self.transform = transform
        self.test_transform = test_transform
        self.mode = 'train'

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, idx):
        name = self.all_names[idx]
        try:
            data = self.name2cache[name]
        except KeyError:
            data = self.read_data(name)
            self.name2cache[name] = data
        data = self.transform(data)
        return {'names': name, 'images': data, 'labels': self.class_name}

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'

    def transform(self, x):
        if self.mode == 'train':
            out = self.train_transform(x) if self.train_transform else x
            return out
        else:
            out = self.test_transform(x) if self.test_transform else x
            return out

    def read_data(self, name):
        path = os.path.join(self.root, 'all', self.class_name, name)
        if self.data_type == 'images':
            return Image.open(path + '.jpg').convert('RGB')
        else:
            return np.load(path + '.npy')


class MiniImagenetFewshotSet(object):
    def __init__(self, root, split_file, n_train, n_test, n_way,
                 length, data_type='images', n_copy=1,
                 transform=None, test_transform=None,
                 memoization=False):
        self.root = root
        self.n_train, self.n_test = n_train, n_test
        self.n_copy = n_copy
        self.length = length
        self.n_way = n_way
        self.train_transform = transform
        self.test_transform = test_transform if test_transform else transform
        self.memoization = memoization
        self.mode = 'train'

        self.classes, self.filenames = self._get_metadata(split_file)
        self.all_classes = sorted(list(set(self.classes)))
        assert len(self.filenames) == len(self.classes)

        self.name2cache = {}
        self.class_to_names = {c: [] for c in self.classes}
        for i in range(len(self.filenames)):
            self.class_to_names[self.classes[i]].append(self.filenames[i])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        task_classes = random.sample(self.all_classes, self.n_way)
        train_samples_list = []
        test_samples_list = []
        for cat_id, cat_name in enumerate(task_classes):
            all_class_inds = self.class_to_names[cat_name]
            chosen_class_inds = random.sample(
                all_class_inds, self.n_train + self.n_test)

            self.mode = 'train'
            for i, img_ind in enumerate(chosen_class_inds):
                if i == self.n_train:
                    self.mode = 'eval'

                for _ in range(self.n_copy):
                    sample = self.get_one_sample(img_ind, cat_name)
                    sample['labels'] = cat_id
                    if i < self.n_train:
                        train_samples_list.append(sample)
                    else:
                        test_samples_list.append(sample)
        train_samples = merge_dicts(train_samples_list)
        test_samples = merge_dicts(test_samples_list)
        return {'train': train_samples, 'test': test_samples}

    def transform(self, x):
        if self.mode == 'train':
            out = self.train_transform(x) if self.train_transform else x
            return out
        else:
            out = self.test_transform(x) if self.test_transform else x
            return out

    def _get_metadata(self, split_file):
        if isinstance(split_file, list):
            annots = pd.concat(pd.read_csv(s) for s in split_file)
        else:
            annots = pd.read_csv(split_file)
        return (annots['label'].values.tolist(),
                annots['filename'].values.tolist())

    def get_one_sample(self, name, class_name):
        if self.memoization:
            try:
                img = self.name2cache[name]
            except KeyError:
                path = os.path.join(self.root, 'all', class_name, name)
                img = Image.open(path).convert('RGB')
        else:
            path = os.path.join(self.root, 'all', class_name, name)
            img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return {'names': name, 'images': img, 'labels': None}
