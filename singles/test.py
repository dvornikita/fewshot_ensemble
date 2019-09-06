#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm
from functools import partial
import local_setup

from data.data_utils import (get_mini_imagenet_transform_augmented,
                             get_cub_transform_augmented, sample_to_device)
from data.meta_datasets import MiniImagenetFewshotSet
from data.dataset_lmdb import LMDBMetaDataset
from torch.utils.data import DataLoader
from models.resnets import resnet18
from models.shallow_net import ShallowNet
from models.losses import get_val_loss_fn
from models.utils import CheckPointer
from singles.config import args
from paths import MINI_IMAGENET_ROOT, CUB_ROOT, TIERED_IMAGENET_ROOT
from utils import print_res, device, set_determ


def main():
    if 'imagenet' in args['data.dataset']:
        root = (TIERED_IMAGENET_ROOT if 'tiered'
                in args['data.dataset'] else MINI_IMAGENET_ROOT)
        imgs_path = os.path.join(root, 'images')
        annot_file_template = os.path.join(root, '{}.csv')
        transform_fn = get_mini_imagenet_transform_augmented
    else:
        imgs_path = os.path.join(CUB_ROOT, 'images')
        annot_file_template = os.path.join(CUB_ROOT, '{}.csv')
        transform_fn = get_cub_transform_augmented

    aug_dict = {key[4:]: args[key] for key in args if key.startswith('aug.')}
    transform_template = partial(transform_fn, aug_dict=aug_dict,
                                 backbone=args['model.backbone'])

    support_mode = 'train' if args['test.augment_support'] else 'test'
    support_transform = transform_template(mode=support_mode)

    query_mode = 'train' if args['test.augment_query'] else 'test'
    query_transform = transform_template(mode=query_mode)

    if not args['data.lmdb']:
        test_annot = annot_file_template.format(args['test.set'])
        testset = MiniImagenetFewshotSet(imgs_path, test_annot, args['data.test_shot'],
                                         args['data.test_query'], args['data.test_way'],
                                         n_copy=args['test.n_copy'], length=args['test.size'],
                                         transform=support_transform,
                                         test_transform=query_transform)
    else:
        testset = LMDBMetaDataset(os.path.join(root, 'lmdb', args['test.set']),
                                  args['data.test_shot'], args['data.test_query'],
                                  args['data.test_way'], args['test.size'],
                                  transform=support_transform)

    test_sampler = DataLoader(testset, batch_size=1, num_workers=args['data.num_workers'])


    # loading the model
    if 'deep' in args['model.backbone']:
        model_fn = resnet18
    elif args['model.backbone'] == 'wide':
        from models.wide_resnet import wide_resnet
        model_fn = wide_resnet
    else:
        model_fn = partial(ShallowNet, args['model.x_dim'],
                           args['model.hid_dim'],
                           args['model.z_dim'])

    # getting number of classes in the training set
    if 'imagenet' in args['model.dataset']:
        ncls = 351 if 'tiered' in args['model.dataset'] else 64
    else:
        ncls = 100

    model = model_fn(classifier=args['model.classifier'],
                     num_classes=ncls)
                     # num_classes=84)
    model.to(device)
    model.eval()

    checkpointer = CheckPointer('singles', args, model)
    ckpt_path = checkpointer.best_ckpt
    if os.path.isfile(checkpointer.best_ckpt) and args['train.resume']:
        start_epoch, best_val_loss, best_val_acc, waiting_for =\
            checkpointer.restore_model(ckpt='best', strict=False)
    else:
        print('Model ckpt file was not found at {}'.format(ckpt_path))
        exit(1)

    if args['test.pca_dim']:
        from classic.utils import get_pca_transform
        args['transform'] = get_pca_transform(model, transform_template)
    loss = get_val_loss_fn(args)

    if args['test.determ']:
        set_determ(args['test.seed'])
    losses, accs, preds = [], [], []
    try:
        for sample in tqdm(test_sampler):
            with torch.no_grad():
                _, stats_dict, pred_dict = loss(model, sample, args['test.distance'])
            losses.append(stats_dict['loss'])
            accs.append(stats_dict['acc'] * 100)
            if args['test.dump_preds']:
                preds.append(pred_dict['preds'])
    except RuntimeError:
        pass

    model_path = '/'.join(ckpt_path.split('/')[:-1])
    print_res(losses, 'loss', os.path.join(model_path, 'stats.txt'))
    print_res(accs, 'accuracy', os.path.join(model_path, 'stats.txt'), prec=2)
    if args['test.dump_preds']:
        import numpy as np
        np.save(os.path.join(model_path, 'preds'), np.array(preds))


if __name__ == '__main__':
    main()
