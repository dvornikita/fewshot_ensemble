import os
import json
import lmdb
import torch
import pickle as pkl

from tqdm import tqdm
from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader

from data.datasets import MiniImagenetDataset
from data.meta_datasets import MiniImagenetFewshotSet
from paths import MINI_IMAGENET_ROOT, CUB_ROOT, TIERED_IMAGENET_ROOT


class LMDBDataset:
    def __init__(self, path, transform=None, test_transform=None,
                 class_id=None, verbose=True):
        self._path = path
        self.mode = 'train'
        self._env = None
        self._txn = None
        self.keys = None

        info = json.load(open(os.path.join(path, 'info.json')))
        self.filenames = info['filenames']
        self.all_classes = info['all_labels']
        self.ids_to_cats = info['ids_to_cats']
        self.labels2keys = info['labels2keys']
        self.cats_to_ids = {value: key for key, value in self.ids_to_cats.items()}

        self._channels = []
        self._load_db(info, class_id)
        self.train_transform = transform
        self.test_transform = test_transform if test_transform else transform
        if class_id is not None:
            self.all_classes = [0]
            self.filenames = self.keys

        if verbose:
            print('Created a dataset of {} categories, with {} images'.format(
                len(self.all_classes), len(self.filenames)))

    def __len__(self):
        return len(self.keys)

    # return stacked frames of shape NxHxW
    # for each color channel where N is the slice size
    def __getitem__(self, idx):
        buffer = self._txn.get(self.keys[idx])
        sample = pkl.load(BytesIO(buffer))
        sample['images'] = Image.open(sample['images']).copy()
        if self.train_transform is not None or self.test_transform is not None:
            sample['images'] = self.transform(sample['images'])
            sample['labels'] = torch.tensor(sample['labels'])
        return sample

    # open lmdb environment and transaction
    # load keys from cache
    def _load_db(self, info, class_id):
        path = self._path

        self._env = lmdb.open(
            self._path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self._txn = self._env.begin(write=False)

        if class_id is None:
            cache_file = os.path.join(path, 'keys')
            if os.path.isfile(cache_file):
                self.keys = pkl.load(open(cache_file, 'rb'))
            else:
                print('Loading dataset keys...')
                with self._env.begin(write=False) as txn:
                    self.keys = [key.decode('ascii')
                                for key, _ in tqdm(txn.cursor())]
                pkl.dump(self.keys, open(cache_file, 'wb'))
        else:
            self.keys = [str(k).encode() for k in info['labels2keys'][str(class_id)]]

        if not self.keys:
            raise ValueError('Empty dataset.')

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


class LMDBMetaDataset_old(MiniImagenetFewshotSet):
    def __init__(self, path, n_train, n_test, n_way, length, n_copy=1,
                 transform=None, test_transform=None):
        self.n_train, self.n_test = n_train, n_test
        self.n_copy = n_copy
        self.length = length
        self.n_way = n_way
        self.train_transform = transform
        self.test_transform = test_transform if test_transform else transform
        info = json.load(open(os.path.join(path, 'info.json')))
        self.all_classes = info['all_labels']
        self.class2set = self.init_class_sets(path)

    def init_class_sets(self, path):
        class2set = dict()
        total_images = 0
        for class_id in self.all_classes:
            class2set[class_id] = LMDBDataset(path, class_id=class_id,
                                              transform=self.train_transform,
                                              verbose=False)
            total_images += len(class2set[class_id].filenames)
        print('Created a Meta dataset of {} categories, with {} images'.format(
            len(self.all_classes), total_images))
        return class2set


class LMDBMetaDataset(MiniImagenetFewshotSet):
    def __init__(self, path, n_train, n_test, n_way, length, n_copy=1,
                 transform=None, test_transform=None):
        self.n_train, self.n_test = n_train, n_test
        self.n_copy = n_copy
        self.length = length
        self.n_way = n_way
        self.train_transform = transform
        self.test_transform = test_transform if test_transform else transform
        self.full_dataset = LMDBDataset(path, transform=transform,
                                        test_transform=test_transform)
        self.classes, self.filenames, self.class_to_names = self._get_metadata()
        self.all_classes = list(self.class_to_names.keys())

    def _get_metadata(self):
        class_to_keys = {self.full_dataset.ids_to_cats[key]:
                         [str(d).encode() for d in val] for key, val
                         in self.full_dataset.labels2keys.items()}
        return None, self.full_dataset.filenames, class_to_keys

    def get_one_sample(self, idx, class_name):
        sample = self.full_dataset[int(idx)]
        sample['labels'] = class_name
        sample['names'] = self.filenames[int(idx)]
        return sample


class DatasetWriter(object):
    def __init__(self, path, dataset, rewrite=False, write_frequency=10000):
        self._write_frequency = write_frequency
        self._path = path
        self._db = None
        if not (os.path.exists(path)):
            os.mkdir(path)
        self._keys_file = os.path.join(path, 'keys')
        self._keys = []
        self._labels2keys = {l: [] for l in dataset.all_labels}

        self._data_loader = DataLoader(dataset, num_workers=16,
                                       collate_fn=lambda x: x)
        self._map_size = len(dataset) * 224 ** 2 * 256

        self._info = dict(
            all_labels=dataset.all_labels,
            ids_to_cats=dataset.ids_to_cats,
            filenames=dataset.filenames)
        self._info_path = os.path.join(path, 'info.json')

        if os.path.exists(self._info_path) and not rewrite:
            raise NameError("Dataset {} already exists.".format(self._path))

    # do not initialize during __init__ to avoid pickling error when using MPI
    def init(self):
        self._db = lmdb.open(self._path, map_size=self._map_size, map_async=True)

    def close(self):
        self.write_keys_and_info()
        if self._db is not None:
            self._db.sync()
            self._db.close()
            self._db = None

    def write_keys_and_info(self):
        with open(self._info_path, 'w') as f:
            json.dump(self._info, f)

        keys = tuple(self._keys)
        pkl.dump(keys, open(self._keys_file, 'wb'))

    def encode_dataset(self):
        if self._db is None:
            self.init()

        txn = self._db.begin(write=True)
        for idx, sample in enumerate(tqdm(self._data_loader)):
            sample = sample[0]

            # compressing image
            img_buffer = BytesIO()
            sample['images'].save(img_buffer, 'JPEG')
            sample['images'] = img_buffer

            # preparing key and adding to lists, dicts
            key = u'{}'.format(idx).encode('ascii')
            self._keys.append(key)
            self._labels2keys[sample['labels']].append(idx)

            # serializing sample and putting in lmdb
            buffer = BytesIO()
            pkl.dump(sample, buffer)
            txn.put(key, buffer.getvalue())

            # flushing into lmdb
            if idx > 0 and idx % self._write_frequency == 0:
                print("[%d/%d]" % (idx, len(self._data_loader)))
                txn.commit()
                txn = self._db.begin(write=True)
        txn.commit()

        self._info['labels2keys'] = self._labels2keys


if __name__ == '__main__':
    test_mode = True
    write = True

    if test_mode:
        dataset = LMDBDataset(os.path.join(
            TIERED_IMAGENET_ROOT, 'lmdb', 'train'))
        import pudb; pudb.set_trace()
    else:
        # dataset_name = 'tiered_imagenet'
        # split = 'test'
        for dataset_name in ['imagenet', 'tiered_imagenet']:
            for split in ['train', 'val', 'test']:
                if 'imagenet' in dataset_name:
                    root = (TIERED_IMAGENET_ROOT if 'tiered'
                            in dataset_name else MINI_IMAGENET_ROOT)
                    imgs_path = os.path.join(root, 'images')
                    annot_file_template = os.path.join(root, '{}.csv')
                else:
                    imgs_path = os.path.join(CUB_ROOT, 'images')
                    annot_file_template = os.path.join(CUB_ROOT, '{}.csv')

                annots = annot_file_template.format(split)
                dataset = MiniImagenetDataset(imgs_path, annots, transform=None,
                                            memoization=False)

                if write:
                    dr = DatasetWriter(os.path.join(root, 'lmdb', split), dataset, True)
                    dr.init()
                    dr.encode_dataset()
                    dr.close()

                    print('Done: {}-{}'.format(dataset_name, split))
                else:
                    path = os.path.join(MINI_IMAGENET_ROOT, 'lmdb', split)
                    lmdb_dataset = LMDBDataset(path)
                    import pudb; pudb.set_trace()

