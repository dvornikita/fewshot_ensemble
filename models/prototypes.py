import torch
import torch.nn.functional as F
from utils import device


def compute_prototypes(net, sample, n_shot, n_way, n_copy=1, regular=True):
    images = sample['images'].squeeze()
    embed_batch = net.embed(images)
    dim = embed_batch.shape[-1]
    if regular:
        assert images.shape[0] == n_shot * n_way * n_copy
        prots = embed_batch.reshape([n_way, n_shot * n_copy, dim])
        prots = torch.mean(prots, dim=1)
    else:
        labels = sample['labels'].squeeze()
        prots = torch.zeros(n_way, dim).type(torch.float)
        for i in range(n_way):
            prots[i] = embed_batch[(labels == i).nonzero(), :].mean(0)
    return prots


def protoloss(prototypes, embeddings, targets, dist='l2'):
    prots = prototypes.unsqueeze(0)
    embeds = embeddings.unsqueeze(1)

    if dist == 'l2':
        scores = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    if dist == 'cos':
        scores = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 20
    if dist == 'lin':
        scores = torch.einsum('izd,zjd->ij',embeds, prots)
    preds = scores.argmax(-1)
    labels = targets.type(torch.long)
    acc_val = torch.eq(preds, labels).float().mean()

    log_p_y = F.log_softmax(scores, dim=1)
    loss_val = F.nll_loss(log_p_y, labels, reduction='mean')
    stats_dict = {'loss': loss_val.item(), 'acc': acc_val.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy(),
                 'probs': torch.softmax(scores, -1)}

    return loss_val, stats_dict, pred_dict


def compute_prototypes_ensemble(net, sample, n_shot, n_way, n_copy=1, regular=True):
    images = sample['images'].squeeze()
    embeds_list = net.embed(images)
    prots_list = []
    for embeds in embeds_list:
        dim = embeds.shape[-1]
        if regular:
            assert images.shape[0] == n_shot * n_way * n_copy
            prots = embeds.reshape([n_way, n_shot * n_copy, dim])
            prots = torch.mean(prots, dim=1)
        else:
            labels = sample['labels'].squeeze()
            prots = torch.zeros(n_way, dim).type(torch.float).to(device)
            for i in range(n_way):
                prots[i] = embeds[(labels == i).nonzero(), :].mean(0)
        prots_list.append(prots)
    return prots_list


def protoloss_ensemble(prototypes, embeddings, targets, dist='l2'):
    prots = prototypes.unsqueeze(0)
    embeds = embeddings.unsqueeze(1)

    if dist == 'l2':
        scores = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    if dist == 'cos':
        scores = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 20
    preds = scores.argmax(-1)
    labels = targets.type(torch.long)
    acc_val = torch.eq(preds, labels).float().mean()

    log_p_y = F.log_softmax(scores, dim=1)
    loss_val = F.nll_loss(log_p_y, labels, reduction='mean')
    stats_dict = {'loss': loss_val.item(), 'acc': acc_val.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy(),
                 'probs': torch.softmax(scores, dim=-1).cpu().numpy()}

    return loss_val, stats_dict, pred_dict
