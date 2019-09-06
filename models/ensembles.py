import torch

from functools import partial
import contextlib
from config import args
from utils import device


def sample_bool_inds(n_heads, p):
    bool_inds = (torch.rand(n_heads) < p).type(torch.uint8)
    if bool_inds.type(torch.int).sum() == 0:
        rnd_ind = (torch.rand(1) * n_heads).type(torch.long)
        bool_inds[rnd_ind] = 1
    return bool_inds


def relation_loss(logits, labels, reg_type, T):
    batch_size, n_cats, n_heads = logits.shape
    if n_heads < 2:
        return 0
    all_probs = torch.softmax(logits / T, dim=1)
    label_inds = torch.ones(batch_size, n_cats).to(device)
    label_inds[range(batch_size), labels] = 0

    # removing the gt prob
    probs = all_probs * label_inds.unsqueeze(-1).detach()
    # re-normalize such that probs sum to 1
    probs /= (all_probs.sum(dim=1, keepdim=True) + 1e-8)

    if 'l2' in reg_type:
        dist_mat = probs.unsqueeze(-1) - probs.unsqueeze(-2)
        dist_mat = dist_mat ** 2
        den = batch_size * (n_heads - 1) * n_heads
        loss = dist_mat.sum() / den
    elif 'cos' in reg_type:
        probs = probs / torch.sqrt(((
            all_probs ** 2).sum(dim=1, keepdim=True) + 1e-8))    # l2 normed
        cov_mat = torch.einsum('ijk,ijl->ikl', probs, probs)
        pairwise_inds = 1 - torch.eye(n_heads).to(device)
        den = batch_size * (n_heads - 1) * n_heads
        loss = (cov_mat * pairwise_inds).sum() / den
    elif 'js' in reg_type:
        loss, count = 0.0, 0
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                p1, p2 = probs[:, :, i], probs[:, :, j]
                interm = (p1 * torch.log(p2 + 1e-7)
                          + p2 * torch.log(p1 + 1e-7)) / 2
                count += 1
                loss -= interm.mean(0).sum()
        loss = loss / count
    elif 'distil':
        loss, count = 0, 0
        summed_targets = all_probs.sum(-1)
        for i in range(n_heads):
            targets = (summed_targets - all_probs[:, :, i]) / (n_heads - 1)
            tlogits = logits[:, :, i] / T
            max_tlogits = tlogits.max(dim=-1, keepdim=True)[0].detach()
            dif_tlogits = tlogits - max_tlogits
            log_prob = tlogits - max_tlogits - torch.log(
                torch.exp(dif_tlogits).sum(-1, keepdim=True))

            loss = loss - (targets * log_prob).sum(-1).mean()
            count += 1
        loss = loss / count

    if 'neg' in reg_type:
        loss = -loss
    return loss


class Ensemble(object):
    def __init__(self, model_fn, num_heads=6, num_classes=64,
                 prob=1, dropout=0):
        self.models = [model_fn(classifier=args['model.classifier'],
                                num_classes=num_classes,
                                dropout=dropout).to(device)
                       for i in range(num_heads)]
        self.prob = prob
        self.num_heads = num_heads
        self.train()

    def embed(self, x, avg=False):
        outs = []
        with self.maybe_grad():
            bool_inds = sample_bool_inds(self.num_heads, self.prob)
            for i in range(self.num_heads):
                if not bool_inds[i] and self._train:
                    continue
                emb_out = self.models[i].embed(x)
                if isinstance(emb_out, list):
                    outs.extend(emb_out)
                else:
                    outs.append(emb_out)
            output = sum(outs) / len(outs) if avg else outs
        return output

    def forward(self, x, avg=False, softmax=False, T=1):
        outs = []
        if not isinstance(x, (list, tuple)):
            x = [x for _ in range(self.num_heads)]
        bool_inds = sample_bool_inds(self.num_heads, self.prob)
        for i in range(self.num_heads):
            if not bool_inds[i] and self._train:
                continue
            out = self.models[i].forward(x[i])
            if softmax:
                out = torch.softmax(out / T, -1)
            outs.append(out)

        output = sum(outs) / len(outs) if avg else outs
        return output

    def train(self):
        self._train = True
        self.maybe_grad = contextlib.suppress
        for i in range(len(self.models)):
            self.models[i].train()

    def eval(self):
        self._train = False
        self.maybe_grad = torch.no_grad
        for i in range(len(self.models)):
            self.models[i].eval()

    def to(self, device):
        for i in range(len(self.models)):
            self.models[i].to(device)

    def parameters(self):
        for i in range(len(self.models)):
            for p in self.models[i].parameters():
                yield p


class Distilation(object):
    def __init__(self, ensemble, T):
        self.ensemble = ensemble
        self.T = T

    def get_loss(self, x, logits):
        with torch.no_grad():
            targets = self.ensemble.forward(x, T=self.T)
        targets = sum(targets) / len(targets)

        tlogits = logits / self.T
        max_tlogits = tlogits.max(dim=-1, keepdim=True)[0].detach()
        dif_tlogits = tlogits - max_tlogits
        log_prob = tlogits - max_tlogits - torch.log(
            torch.exp(dif_tlogits).sum(-1, keepdim=True))

        loss = - (targets * log_prob).sum(-1).mean()
        return loss

