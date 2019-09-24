#!/usr/bin/env python3

import os
import torch
import numpy as np
from tqdm import tqdm
import local_setup

from data.helpers import get_test_sampler
from models.losses import get_val_loss_fn_ensemble as get_val_loss_fn
from models.utils import EnsembleCheckPointer as CheckPointer
from models.helpers import get_model
from ensembles.config import args
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
    model = get_model(ncls, args, ensemble=True)
    model.eval()

    checkpointer = CheckPointer('ensembles', args, model)
    ckpt_path = checkpointer.best_ckpt
    if os.path.isfile(ckpt_path):
        start_epoch, best_val_loss, best_val_acc, waiting_for =\
            checkpointer.restore_model(ckpt='best', strict=False)
    elif 'copy__' in ckpt_path:
        checkpointer.restore_from_singles()
    else:
        print('Model ckpt file was not found at {}'.format(ckpt_path))
        exit(1)


    # to ensure deterministic behavior
    if args['test.determ']:
        set_determ(args['test.seed'])

    # to collect testing results and report statistics later
    losses, probsum_accs, voted_accs, all_accs, agreements = [], [], [], [], []

    for sample in tqdm(test_sampler):
        with torch.no_grad():
            _, stats_dict, pred_dict = loss(model, sample)
        losses.append(stats_dict['loss'])
        voted_accs.append(stats_dict['voted_acc'] * 100)
        probsum_accs.append(stats_dict['probsum_acc'] * 100)
        all_accs.append(stats_dict['accs'] * 100)
        agreements.append(stats_dict['agreement'])

    model_path = '/'.join(ckpt_path.split('/')[:-1])
    stats_path = os.path.join(model_path, 'stats.txt')
    if 'copy__' in model_path or args['test.noprint']:
        stats_path = None

    # processing and printing out statistics
    agreement_mat = np.array(agreements).mean(0)
    np.set_printoptions(precision=2)
    print(agreement_mat)

    accs_mat = np.stack(all_accs)
    accs = accs_mat.mean(0)
    min_ind = accs.argmin(0)
    max_ind = accs.argmax(0)
    print('All accs: ', accs)
    print_res(losses, 'loss', stats_path)
    print_res(accs_mat.mean(1), 'mean_accuracy', stats_path, prec=2)
    print_res(accs_mat[:, min_ind], 'min_accuracy', stats_path, prec=2)
    print_res(accs_mat[:, max_ind], 'max_accuracy', stats_path, prec=2)
    print_res(voted_accs, 'voted_accuracy', stats_path, prec=2)
    print_res(probsum_accs, 'probsum_accuracy', stats_path, prec=2)


if __name__ == '__main__':
    main()
