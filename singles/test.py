#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm
import local_setup

from data.helpers import get_test_sampler
from models.losses import get_val_loss_fn
from models.utils import CheckPointer
from models.helpers import get_model
from singles.config import args
from utils import print_res, set_determ


def main():
    testset, test_sampler = get_test_sampler(args)
    loss = get_val_loss_fn(args)

    # getting number of classes in the training set
    if 'imagenet' in args['model.dataset']:
        ncls = 351 if 'tiered' in args['model.dataset'] else 64
    else:
        ncls = 100

    # defining the model
    model = get_model(ncls, args, ensemble=False)
    model.eval()

    checkpointer = CheckPointer('singles', args, model)
    ckpt_path = checkpointer.best_ckpt
    if os.path.isfile(ckpt_path):
        start_epoch, best_val_loss, best_val_acc, waiting_for =\
            checkpointer.restore_model(ckpt='best', strict=False)
    else:
        print('Model ckpt file was not found at {}'.format(ckpt_path))
        exit(1)

    loss = get_val_loss_fn(args)

    if args['test.determ']:
        set_determ(args['test.seed'])

    losses, accs, _ = [], [], []
    for sample in tqdm(test_sampler):
        with torch.no_grad():
            _, stats_dict, pred_dict = loss(model, sample, args['test.distance'])
        losses.append(stats_dict['loss'])
        accs.append(stats_dict['acc'] * 100)

    model_path = '/'.join(ckpt_path.split('/')[:-1])
    print_res(losses, 'loss', os.path.join(model_path, 'stats.txt'))
    print_res(accs, 'accuracy', os.path.join(model_path, 'stats.txt'), prec=2)


if __name__ == '__main__':
    main()
