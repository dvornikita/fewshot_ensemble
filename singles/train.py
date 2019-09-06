#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm
from functools import partial

import local_setup
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from data.data_utils import sample_to_device, unnormalize
from data.helpers import get_trainval_samplers
from models.losses import get_train_loss_fn, get_val_loss_fn
from models.utils import CheckPointer
from models.helpers import get_model, get_optimizer
from singles.config import args
from utils import Accumulator, device, set_determ

if args['train.determ']:
    set_determ


def main():
    trainset, valset, train_sampler, val_sampler = get_trainval_samplers(args)
    s2d = partial(sample_to_device, device=device)

    train_loss_fn = get_train_loss_fn(args)
    val_loss_fn = get_val_loss_fn(args)

    # Defining the model and Restoring the last checkpoint
    model = get_model(len(trainset.all_classes),
                      args, ensemble=False)
    optimizer = get_optimizer(model, args)

    # Restoring the last checkpoint
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

    n_train, n_val = len(train_sampler), len(val_sampler)
    epoch_loss = Accumulator(n_train)
    epoch_acc = Accumulator(n_train)
    for epoch in range(start_epoch, args['train.epochs']):
        print('\n !!!!!! Starting ephoch %d !!!!!!' % epoch)

        model.train()
        for i, sample in enumerate(tqdm(train_sampler)):
            optimizer.zero_grad()

            sample = s2d(sample)
            logits = model.forward(sample['images'])
            batch_loss, stats_dict, _ = train_loss_fn(logits, sample['labels'])
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
            # write images
            if args['train.image_summary'] and t % 1000 == 0:
                grid = make_grid(unnormalize(sample['images'][:9]))
                writer.add_image('images', grid, t)

        # since cub is smaller, evaluate every 5 epochs
        if args['data.dataset'] == 'cub' and epoch % 5 != 0:
            continue

        model.eval()
        val_loss, val_acc = Accumulator(n_val), Accumulator(n_val)
        evaled, evi, totalcount = False, 0, 0
        while totalcount < len(val_sampler) and evi < 5:
            try:
                for j, val_sample in enumerate(tqdm(val_sampler)):
                    with torch.no_grad():
                        _, stats_dict, _ = val_loss_fn(model, val_sample)
                    val_loss.append(stats_dict['loss'])
                    val_acc.append(stats_dict['acc'])
                    totalcount += 1
                evaled = True
            except RuntimeError:
                evi += 1
                print('Not evaled')
        assert evaled, 'Not Evaluated!'

        print('train_loss: {0:.4f}, train_acc {1:.2f}%, val_loss: {2:.4f}, val_acc {3:.2f}%'.format(
            epoch_loss.mean(), epoch_acc.mean() * 100, val_loss.mean(), val_acc.mean() * 100))

        # write summaries
        writer.add_scalar('loss/val_loss', val_loss.mean(), epoch)
        writer.add_scalar('accuracy/val_acc', val_acc.mean() * 100, epoch)

        if val_loss.mean() < best_val_loss:
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
            checkpointer.save_checkpoint(epoch, best_val_acc, best_val_loss,
                                         waiting_for, is_best, optimizer=optimizer)

    writer.close()
    print('\n Done with train_loss: {0:.4f}, train_acc {1:.2f}%, val_loss: {2:.4f}, val_acc {3:.2f}%'
          .format(best_train_loss, best_train_acc * 100, best_val_loss, best_val_acc * 100))


if __name__ == '__main__':
    main()
