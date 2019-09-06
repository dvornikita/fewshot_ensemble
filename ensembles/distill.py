#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm
from functools import partial

import local_setup
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data.datasets import CocoDataset
from data.data_utils import sample_to_device
from data.helpers import get_trainval_samplers, init_dataset
from models.ensembles import Distilation
from models.losses import get_train_loss_fn, get_val_loss_fn
from models.utils import CheckPointer
from models.helpers import get_model, get_optimizer
from ensembles.config import args
from utils import Accumulator, device, set_determ

if args['train.determ']:
    set_determ


def main():
    batch_size = args['data.batch_size']
    _, transform_template = init_dataset(args)
    trainset, valset, train_sampler, val_sampler = get_trainval_samplers(args)
    s2d = partial(sample_to_device, device=device)

    # if distilling with extra data
    if args['distil.unsup_size'] > 0:
        distilset = CocoDataset(args['distil.unsup_set'],
                                transform=transform_template(mode='train'))
        distil_sampler = DataLoader(distilset, batch_size=args['distil.unsup_size'],
                                    shuffle=True, num_workers=args['data.num_workers'])

    val_loss_fn = get_val_loss_fn(args)
    train_loss_fn = get_train_loss_fn(args)

    # defining the model
    model = get_model(len(trainset.all_classes),
                      args, ensemble=False)
    ensemble = get_model(len(trainset.all_classes),
                         args, ensemble=True)
    ensemble.train()
    distil = Distilation(ensemble, T=args['distil.T'])
    optimizer = get_optimizer(model, args)

    checkpointer = CheckPointer('singles', args, model, optimizer=optimizer)
    if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
        start_epoch, best_val_loss, best_val_acc, waiting_for =\
            checkpointer.restore_model(ckpt='last')
    else:
        print('No checkpoint restoration')
        best_val_loss = 999999999
        best_val_acc = waiting_for = start_epoch = 0

    # defining the summary writer
    writer = SummaryWriter(checkpointer.model_path)

    gamma = args['distil.gamma']
    unsup = args['distil.unsup_size'] > 0
    n_train, n_val = len(train_sampler), len(val_sampler)
    epoch_loss = Accumulator(n_train)
    epoch_distil = Accumulator(n_train)
    epoch_acc = Accumulator(n_train)
    for epoch in range(start_epoch, args['train.epochs']):
        print('\n !!!!!! Starting ephoch %d !!!!!!' % epoch)

        if not unsup:
            distil_sampler = range(len(train_sampler))

        model.train()
        for i, (dist_sample, sample) in enumerate(zip(distil_sampler, tqdm(train_sampler))):
            optimizer.zero_grad()

            sample = s2d(sample)
            if unsup:
                dist_sample = s2d(dist_sample)
                all_images = torch.cat([sample['images'], dist_sample['images']], 0)
            else:
                all_images = sample['images']

            logits = model.forward(all_images)
            ce_loss, stats_dict, _ = train_loss_fn(logits[:batch_size], sample['labels'])
            distil_loss = distil.get_loss(all_images, logits)
            batch_loss = (1 - gamma) * ce_loss + gamma * distil.T ** 2 * distil_loss

            epoch_distil.append(distil_loss.item())
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
                writer.add_scalar('loss/distil_loss',
                                  epoch_distil.mean(last=100), t)
                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], t)

        if args['data.dataset'] == 'cub' and epoch % 5 != 0:
            continue
        model.eval()
        val_loss, val_acc = Accumulator(n_val), Accumulator(n_val)
        for j, val_sample in enumerate(tqdm(val_sampler)):
            with torch.no_grad():
                _, stats_dict, _ = val_loss_fn(model, val_sample)
            val_loss.append(stats_dict['loss'])
            val_acc.append(stats_dict['acc'])

        print('train_loss: {0:.4f}, train_acc {1:.2f}%, val_loss: {2:.4f}, val_acc {3:.2f}%'.format(
            epoch_loss.mean(), epoch_acc.mean() * 100, val_loss.mean(), val_acc.mean() * 100))

        # write summaries
        writer.add_scalar('loss/val_loss', val_loss.mean(), epoch)
        writer.add_scalar('accuracy/val_acc', val_acc.mean() * 100, epoch)

        if val_acc.mean() > best_val_acc:
            best_val_loss = val_loss.mean()
            best_train_loss = epoch_loss.mean()
            best_val_acc = val_acc.mean()
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
            extra = {'distil_name': args['distil.name']}
            checkpointer.save_checkpoint(epoch, best_val_acc, best_val_loss,
                                         waiting_for, is_best,
                                         optimizer=optimizer,
                                         extra=extra)

    writer.close()
    print('\n Done with train_loss: {0:.4f}, train_acc {1:.2f}%, val_loss: {2:.4f}, val_acc {3:.2f}%'.format(best_train_loss, best_train_acc * 100, best_val_loss, best_val_acc * 100))


if __name__ == '__main__':
    main()
