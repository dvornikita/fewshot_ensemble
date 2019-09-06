import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize


MaxPool2d = nn.MaxPool2d
AvgPool2d = nn.AvgPool2d


class Module(nn.Module):
    def init_updts(self):
        for p in self.parameters():
            p.updt = p

    def zero_updts(self):
        for p in self.parameters():
            del p.updt

    def apply_deltas(self, deltas):
        if type(deltas) == list:
            for i, p in enumerate(self.parameters()):
                p.updt = p.updt + deltas[i]
        else:
            # for key, p in self.named_parameters():
            d = dict(self.named_parameters())
            for key in deltas.keys():
                p = d[key]
                p.updt = p.updt + deltas[key]

    def update_weights(self):
        for key, p in self.named_parameters():
            p.data.copy_(p.updt.data)

    def update_spectral_vectors(self, name=None, init=False):
        try:
            if init:
                self.reset_reg_parameters()
            else:
                self.update_spectral_norm(self.u, self.v, 1)
        except AttributeError:
            pass

        for mname, module in self.named_children():
            try:
                module.update_spectral_vectors(name, init=init)
            except AttributeError:
                continue

    def get_spectral_loss(self, memo=None, name=None):
        if memo is None:
            memo = set()

        spectral_loss = 0.
        try:
            p = self.weight
            if p is not None and p not in memo:
                memo.add(p)
                spectral_loss += self.get_spectral_reg()
        except AttributeError:
            pass

        for mname, module in self.named_children():
            try:
                spectral_loss += module.get_spectral_loss(memo, mname)
            except AttributeError:
                continue
        return spectral_loss


class Sequential(nn.modules.container.Sequential, Module):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__(*args, **kwargs)


class Linear(nn.Linear, Module):
    def forward(self, input):
        bias = self.bias if self.bias is None else self.bias.updt
        return F.linear(input, self.weight.updt, bias)


class Conv2d(nn.Conv2d, Module):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_reg_parameters(self):
        out_channels, in_channels, ks1, ks2 = self.weight.shape
        h, w = out_channels, in_channels * ks1 * ks2

        try:
            self.left_vec
        except AttributeError:
            self.left_vec = nn.Parameter(torch.Tensor(in_channels, ks1, ks2))
            self.right_vec = nn.Parameter(torch.Tensor(out_channels))

        u = normalize(self.weight.new_empty(h).normal_(0, 1), dim=0, eps=1e-10)
        v = normalize(self.weight.new_empty(w).normal_(0, 1), dim=0, eps=1e-10)
        self.update_spectral_norm(u, v, n_power_iterations=3)

    def update_spectral_norm(self, u, v, n_power_iterations=1):
        weight_mat = self.weight.view(self.weight.shape[0], -1)
        h, w = weight_mat.shape
        with torch.no_grad():
            for _ in range(n_power_iterations):
                v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=1e-10, out=v)
                u = normalize(torch.mv(weight_mat, v), dim=0, eps=1e-10, out=u)
            if n_power_iterations > 0:
                u = u.clone()
                v = v.clone()
        self.left_vec.data.copy_(v.view(self.left_vec.shape).data)
        self.right_vec.data.copy_(u.data)

    def get_spectral_reg(self):
        return torch.einsum('jkl,ijkl,i->', (self.left_vec,
                            self.weight, self.right_vec)) ** 2

    def forward(self, input):
        bias = self.bias if self.bias is None else self.bias.updt
        return F.conv2d(input, self.weight.updt, bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BatchNorm2d(nn.BatchNorm2d, Module):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        bias = self.bias if self.bias is None else self.bias.updt
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight.updt, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class CosineSim(Module):
    def __init__(self, n_feat, num_classes):
        super(CosineSim, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_feat, num_classes))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.init_weights()

    def init_weights(self):
        std = math.sqrt(2.0 / self.weight.size(1))
        self.weight.data.normal_(0.0, std)
        self.bias.data.fill_(1)

    def forward(self, x):
        weight = self.weight.updt
        scaler = self.bias if self.bias is None else self.bias.updt
        return self.cosine_sim(x.unsqueeze(-1), weight.unsqueeze(0)) * scaler

