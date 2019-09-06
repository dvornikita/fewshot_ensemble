#!/usr/bin/env python3

import os
import torch
import numpy as np
from tqdm import tqdm
from functools import partial

import local_setup
from tensorboardX import SummaryWriter
from models.resnets import resnet18
from models.wide_resnet import wide_resnet
from models.ensembles import Ensemble, relation_loss
from models.losses import get_train_loss_fn_ensemble as get_train_loss_fn
from models.losses import get_val_loss_fn_ensemble as get_val_loss_fn
from models.utils import EnsembleCheckPointer as CheckPointer
from models.helpers import get_model, get_optimizer
from ensembles.config import args
from data.helpers import get_trainval_samplers
from data.data_utils import sample_to_device
from utils import Accumulator, device, set_determ


if args['train.determ']:
    set_determ

def main():
    batch_size = args['data.batch_size']
    trainset, valset, train_sampler, val_sampler = get_trainval_samplers(args)
    s2d = partial(sample_to_device, device=device)

    train_loss_fn = get_train_loss_fn(args)
    val_loss_fn = get_val_loss_fn(args)

    # Defining the model and Restoring the last checkpoint
    model = get_model(len(trainset.all_classes),
                      args, ensemble=True)
    optimizer = get_optimizer(model, args)

    checkpointer = CheckPointer(args, model=model, optimizer=optimizer)
    if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
        start_epoch, best_val_loss, best_val_acc, waiting_for =\
            checkpointer.restore_model(ckpt='last')
    else:
        print('No checkpoint restoration')
        best_val_loss = 999999999
        best_val_acc = waiting_for = start_epoch = 0

    # defining the summary writer
    writer = SummaryWriter(checkpointer.model_path)

    n_train, n_val = len(train_sampler), len(val_sampler)
    epoch_loss = Accumulator(n_train)
    epoch_acc = Accumulator(n_train)
    for epoch in range(start_epoch, args['train.epochs']):
        print('\n !!!!!! Starting ephoch %d !!!!!!' % epoch)

        model.train()
        for i, sample in enumerate(tqdm(train_sampler)):
            optimizer.zero_grad()

            sample = s2d(sample)
            if args['ens.robust_matching']:
                images, labels = sample['images'], sample['labels'][:batch_size]
                new_shape = [args['ens.num_heads'], batch_size] + list(images.shape[1:])
                images = images.view(new_shape).unbind(dim=0)
            else:
                images, labels = sample['images'], sample['labels']
            logits_list = model.forward(images)
            labels_list = [labels for _ in range(len(logits_list))]
            batch_loss, stats_dict, _ = train_loss_fn(logits_list, labels_list)
            if args['ens.joint_training'] and epoch >= args['ens.rel_active_epoch']:
                sd_loss = relation_loss(torch.stack(logits_list, -1), labels,
                                        reg_type=args['ens.rel_fn'],
                                        T=args['ens.rel_T'])
                batch_loss += sd_loss * args['ens.rel_coef']
            else:
                sd_loss = 0
            epoch_loss.append(stats_dict['loss'])
            epoch_acc.append(stats_dict['acc'])

            batch_loss.backward()
            optimizer.step()

            t = epoch * n_train + i
            if t % 100 == 0:
                writer.add_scalar('loss/train_loss',
                                  epoch_loss.mean(last=100), t)
                writer.add_scalar('accuracy/train_acc',
                                  epoch_acc.mean(last=100), t)
                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], t)
                if args['ens.joint_training']:
                    writer.add_scalar('loss/softmax_diverse',
                                      sd_loss, t)

        if args['data.dataset'] == 'cub' and epoch % 5 != 0:
            continue

        model.eval()
        evaled, evi, totalcount = False, 0, 0
        while totalcount < len(val_sampler) and evi < 5:
            try:
                val_loss, val_acc = Accumulator(n_val), Accumulator(n_val)
                val_acc_soft = Accumulator(n_val)
                consensuses = []
                for j, val_sample in enumerate(tqdm(val_sampler)):
                    with torch.no_grad():
                        _, stats_dict, _ = val_loss_fn(model, val_sample)
                    val_loss.append(stats_dict['loss'])
                    val_acc.append(stats_dict['voted_acc'])
                    val_acc_soft.append(stats_dict['probsum_acc'])
                    consensuses.append(stats_dict['agreement'])
                    totalcount += 1
                evaled = True
            except RuntimeError:
                evi += 1
                print('Not evaled')
        assert evaled, 'Not Evaluated!'

        print('train_loss: {0:.4f}, train_acc {1:.2f}%, val_loss: {2:.4f}, val_acc {3:.2f}%'.format(
            epoch_loss.mean(), epoch_acc.mean() * 100, val_loss.mean(), val_acc.mean() * 100))

        agreement = np.array(consensuses).mean(0)
        l = agreement.shape[-1]
        agreement -= np.eye(l)
        # write summaries
        writer.add_scalar('loss/val_loss', val_loss.mean(), epoch)
        writer.add_scalar('accuracy/val_acc', val_acc.mean() * 100, epoch)

        writer.add_scalar('accuracy/val_acc_soft',
                          val_acc_soft.mean() * 100, epoch)
        writer.add_scalar('accuracy/_consensus',
                          agreement.sum() / l / (l - 1) * 100, epoch)

        val_acc_ep = val_acc_soft.mean()
        # if val_loss.mean() < best_val_loss:
        if val_acc_ep > best_val_acc:
            best_val_loss = val_loss.mean()
            best_train_loss = epoch_loss.mean()
            best_val_acc = val_acc_ep
            best_train_acc = epoch_acc.mean()
            waiting_for = 0
            is_best = True

            print('Best model so far!')
        else:
            waiting_for += 1
            is_best = False
            if waiting_for >= args['train.patience']:
                mult = args['train.decay_coef']
                print('Decaying lr by the factor of {}'.format(mult))

                # loading the best model so far and optimizing from that point
                checkpointer.restore_model(ckpt='best', model=True)

                for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / mult
                        lr_ = param_group['lr']
                waiting_for = 0
                if args['train.learning_rate'] / lr_ >= mult ** 2 - 0.1:
                    print('Out of patience')
                    break

        # saving checkpoints
        if epoch % args['train.ckpt_freq'] == 0 or is_best:
            checkpointer.save_checkpoint(epoch, best_val_acc, best_val_loss,
                                         waiting_for, is_best, optimizer=optimizer)

    writer.close()
    print('\n Done with train_loss: {0:.4f}, train_acc {1:.2f}%, val_loss: {2:.4f}, val_acc {3:.2f}%'.format(best_train_loss, best_train_acc * 100, best_val_loss, best_val_acc * 100))


if __name__ == '__main__':
    main()
