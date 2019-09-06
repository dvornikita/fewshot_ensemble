import os
from functools import partial

from torch.utils.data import DataLoader
from data.datasets import MiniImagenetDataset
from data.meta_datasets import MiniImagenetFewshotSet as MiniImagenetMeta
from data.dataset_lmdb import LMDBDataset, LMDBMetaDataset
from data.data_utils import (get_mini_imagenet_transform_augmented,
                             get_cub_transform_augmented)
from paths import MINI_IMAGENET_ROOT, CUB_ROOT, TIERED_IMAGENET_ROOT, PROJECT_ROOT


def init_dataset(args):
    if 'imagenet' in args['data.dataset']:
        root = (TIERED_IMAGENET_ROOT if 'tiered'
                in args['data.dataset'] else MINI_IMAGENET_ROOT)
        transform_fn = get_mini_imagenet_transform_augmented
    else:
        root = CUB_ROOT
        transform_fn = get_cub_transform_augmented

    aug_dict = {key[4:]: args[key] for key in args if key.startswith('aug.')}
    transform_template = partial(transform_fn, aug_dict=aug_dict,
                                 backbone=args['model.backbone'])
    return root, transform_template


def get_trainval_samplers(args):
    batch_size = args['data.batch_size']
    num_workers = args['data.num_workers']

    root, transform_template = init_dataset(args)
    imgs_path = os.path.join(root, 'images')
    annot_file_template = os.path.join(PROJECT_ROOT, 'splits',
                                       args['data.dataset'], '{}.csv')

    train_split = args['train.set']
    if not args['data.lmdb']:
        Dataset = MiniImagenetDataset
        # defining the dataset
        if '-' in train_split:
            annots = [annot_file_template.format(s) for s in train_split.split('-')]
        else:
            annots = annot_file_template.format(train_split)
        trainset = Dataset(imgs_path, annots,
                           transform=transform_template(mode='train'))

        valset = MiniImagenetMeta(imgs_path, annot_file_template.format('val'),
                                  args['data.test_shot'], args['data.test_query'],
                                  args['data.test_way'], args['test.size'],
                                  transform=transform_template(mode='test'))
    else:
        path = os.path.join(root, 'lmdb', train_split)
        trainset = LMDBDataset(path, transform_template(mode='train'))
        valset = LMDBMetaDataset(os.path.join(root, 'lmdb', 'val'),
                                 args['data.test_shot'], args['data.test_query'],
                                 args['data.test_way'], args['test.size'],
                                 transform=transform_template(mode='test'))
    if args.get('ens.robust_matching', False):
        from data.data_utils import BatchIdxSampler
        batch_sampler = BatchIdxSampler(trainset, batch_size=batch_size,
                                        n_copy=args['ens.num_heads'])
        train_sampler = DataLoader(trainset, batch_sampler=batch_sampler,
                                   num_workers=num_workers)
    else:
        train_sampler = DataLoader(trainset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)

    val_sampler = DataLoader(valset, batch_size=1,
                             num_workers=args['data.num_workers'])
    return trainset, valset, train_sampler, val_sampler


def get_test_sampler(args):
    num_workers = args['data.num_workers']

    root, transform_template = init_dataset(args)
    imgs_path = os.path.join(root, 'images')
    annot_file_template = os.path.join(PROJECT_ROOT, 'splits',
                                       args['data.dataset'], '{}.csv')

    support_mode = 'train' if args['test.augment_support'] else 'test'
    support_transform = transform_template(mode=support_mode)

    query_mode = 'train' if args['test.augment_query'] else 'test'
    query_transform = transform_template(mode=query_mode)

    if not args['data.lmdb']:
        test_annot = annot_file_template.format(args['test.set'])
        testset = MiniImagenetMeta(imgs_path, test_annot, args['data.test_shot'],
                                   args['data.test_query'], args['data.test_way'],
                                   n_copy=args['test.n_copy'], length=args['test.size'],
                                   transform=support_transform,
                                   test_transform=query_transform)
    else:
        testset = LMDBMetaDataset(os.path.join(root, 'lmdb', args['test.set']),
                                  args['data.test_shot'], args['data.test_query'],
                                  args['data.test_way'], args['test.size'],
                                  transform=support_transform)

    test_sampler = DataLoader(testset, batch_size=1, num_workers=num_workers)
    return testset, test_sampler
