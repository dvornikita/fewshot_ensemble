import numpy as np
import torch
import torch.nn.functional as F

import local_setup
from models.prototypes import (compute_prototypes, protoloss,
                               compute_prototypes_ensemble,
                               protoloss_ensemble)
from utils import process_copies, s2d, voting, agreement


def get_train_loss_fn(args):
    def train_loss_fn(logits, targets):
        log_p_y = F.log_softmax(logits, dim=1)
        preds = log_p_y.argmax(-1)
        labels = targets.type(torch.long)
        loss = F.nll_loss(log_p_y, labels, reduction='mean')
        acc = torch.eq(preds, labels).float().mean()
        stats_dict = {'loss': loss.item(), 'acc': acc.item()}
        pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy(),
                     'log_p': log_p_y.detach().cpu().numpy()}

        return loss, stats_dict, pred_dict
    return train_loss_fn


def get_val_loss_fn(args):
    def val_loss_fn(model, val_sample, dist='cos'):
        train_sample = s2d(val_sample['train'])
        test_sample = s2d(val_sample['test'])

        prototypes = compute_prototypes(model, train_sample,
                                        args['data.test_shot'],
                                        args['data.test_way'],
                                        n_copy=args['test.n_copy'],
                                        regular=True)
        embeddings = model.embed(test_sample['images'].squeeze())
        labels = test_sample['labels'].squeeze()
        if args['test.n_copy'] > 1:
            embeddings, labels = process_copies(
                embeddings, labels, args)
        if 'transform' in args:
            embeddings = args['transform'](embeddings)
            prototypes = args['transform'](prototypes)
            labels = labels.to('cpu')
        loss, stats_dict, pred_dict = protoloss(prototypes, embeddings,
                                                labels, dist=dist)
        return loss, stats_dict, pred_dict
    return val_loss_fn


def get_train_loss_fn_ensemble(args):
    def train_loss_fn_ensemble(logits_list, targets_list):
        total_loss, total_acc = 0, 0
        stats_dict = {'loss': 0, 'acc': 0}
        for logits, targets in zip(logits_list, targets_list):
            if logits is not None:
                log_p_y = F.log_softmax(logits, dim=1)
                preds = log_p_y.argmax(-1)
                labels = targets.type(torch.long)
                loss = F.nll_loss(log_p_y, labels, reduction='mean')
                acc = torch.eq(preds, labels).float().mean()
                total_loss = total_loss + loss
                total_acc = total_acc + acc
        stats_dict['loss'] = total_loss.item()
        stats_dict['acc'] = total_acc.item()
        pred_dict = {'preds': preds.cpu().numpy(),
                     'labels': labels.cpu().numpy()}

        return total_loss, stats_dict, pred_dict
    return train_loss_fn_ensemble


def get_val_loss_fn_ensemble(args):
    def val_loss_fn(model, val_sample, dist='cos'):
        train_sample = s2d(val_sample['train'])
        test_sample = s2d(val_sample['test'])

        prototypes_list = compute_prototypes_ensemble(
            model, train_sample, args['data.test_shot'],
            args['data.test_way'], n_copy=args['test.n_copy'],
            regular=False)
        embeddings_list = model.embed(test_sample['images'].squeeze())
        labels = test_sample['labels'].squeeze()
        losses, accs, preds = [], [], []

        probs = []
        for i in range(len(prototypes_list)):
            prototypes = prototypes_list[i]
            embeddings = embeddings_list[i]
            if args['test.n_copy'] > 1:
                embeddings, labels = process_copies(
                    embeddings, labels, args)
            if 'transform' in args:
                embeddings = args['transform'](embeddings)
                prototypes = args['transform'](prototypes)
                labels = labels.to('cpu')
            loss, stats_dict, pred_dict = protoloss_ensemble(
                prototypes, embeddings, labels, dist=dist)
            probs.append(pred_dict['probs'])
            losses.append(stats_dict['loss'])
            accs.append(stats_dict['acc'])
            preds.append(pred_dict['preds'])
        voted_preds = voting(preds)
        stats_dict['voted_acc'] = (voted_preds == test_sample[
            'labels'].cpu().numpy()).astype('float').mean()
        stats_dict['probsum_acc'] = (
            np.array(probs).mean(0).argmax(-1) == test_sample[
                'labels'].cpu().numpy()).astype('float').mean()
        stats_dict['agreement'] = agreement(np.array(preds))
        stats_dict['loss'] = losses
        stats_dict['accs'] = np.array(accs)
        stats_dict['preds'] = np.array(preds)
        return loss, stats_dict, pred_dict
    return val_loss_fn
