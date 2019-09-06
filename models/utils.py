import os
import torch
import shutil
from torch import nn
from utils import check_dir
from paths import PROJECT_ROOT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class distLinear(nn.Module):
    def __init__(self, n_feat, num_classes):
        super(distLinear, self).__init__()
        self.L = nn.Linear(n_feat, num_classes, bias=False)
        self.scalar = nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.L.reset_parameters()
        self.scalar.data.fill_(1)

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-7)
        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(
            1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 1e-7)
        cos_dist = self.L(x_normalized)
        scores = self.scalar * cos_dist
        return scores


class CheckPointer(object):
    def __init__(self, folder, args, model=None, reg_model=None, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_path = os.path.join(PROJECT_ROOT, folder, 'models',
                                       args['model.dataset'],
                                       args['model.model_name'])
        self.last_ckpt = os.path.join(self.model_path, 'checkpoint.pth.tar')
        self.best_ckpt = os.path.join(self.model_path, 'model_best.pth.tar')

    def restore_model(self, ckpt='last', model=True,
                      optimizer=True, strict=True):
        ckpt_path = self.last_ckpt if ckpt == 'last' else self.best_ckpt

        if os.path.isfile(ckpt_path):
            print("=> loading {} checkpoint '{}'".format(ckpt, ckpt_path))
            ch = torch.load(ckpt_path, map_location=device)
            if self.model is not None and model:
                self.model.load_state_dict(ch['state_dict'], strict=strict)

            if self.optimizer is not None and optimizer:
                self.optimizer.load_state_dict(ch['optimizer'])
            print("=> Loaded! Epoch {}".format(ch['epoch']))
        else:
            assert False, "No checkpoint!"

        return (ch['epoch'], ch['best_val_loss'],
                ch['best_val_acc'], ch['waiting_for'])

    def restore_from_another_model(self, folder, name, model=False,
                                   ckpt='last', strict=True):
        model_path = os.path.join(PROJECT_ROOT, folder, 'models', name)

        ckpt = 'checkpoint' if ckpt == 'last' else 'model_best'
        ckpt_path = os.path.join(model_path, '%s.pth.tar' % ckpt)
        ch = torch.load(ckpt_path, map_location=device)

        if model:
            self.model.load_state_dict(ch['state_dict'], strict=strict)

        print('Restored base model %s' % name)

        return (ch['epoch'], ch['best_val_loss'],
                ch['best_val_acc'], ch['waiting_for'])

    def save_checkpoint(self, epoch, best_val_acc, best_val_loss, waiting_for,
                        is_best, filename='checkpoint.pth.tar', optimizer=None,
                        extra=None):
        state = {'epoch': epoch + 1,
                 'args': self.args,
                 'state_dict': self.model.state_dict(),
                 'best_val_acc': best_val_acc,
                 'best_val_loss': best_val_loss,
                 'waiting_for': waiting_for,
                }

        if extra is not None:
            state.update(extra)

        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        model_path = check_dir(self.model_path, True, False)
        torch.save(state, os.path.join(model_path, filename))
        if is_best:
            shutil.copyfile(os.path.join(model_path, filename),
                            os.path.join(model_path, 'model_best.pth.tar'))


class EnsembleCheckPointer(object):
    def __init__(self, args, model=None, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_path = os.path.join(PROJECT_ROOT, 'ensembles',
                                       'models',
                                       args['model.dataset'],
                                       args['model.model_name'])
        self.last_ckpt = os.path.join(self.model_path, 'checkpoint.pth.tar')
        self.best_ckpt = os.path.join(self.model_path, 'model_best.pth.tar')

    def restore_model(self, ckpt='last', model=True, 
                      optimizer=True, strict=True):
        ckpt_path = self.last_ckpt if ckpt == 'last' else self.best_ckpt

        if os.path.isfile(ckpt_path) or '___' in ckpt_path:
            print("=> loading {} checkpoint '{}'".format(ckpt, ckpt_path))

            ch = torch.load(ckpt_path, map_location=device)

            if self.model is not None and model:
                if not isinstance(self.model, nn.Module):
                    assert len(self.model.models) == len(ch['state_dict'])
                    for i in range(len(ch['state_dict'])):
                        self.model.models[i].load_state_dict(
                            ch['state_dict'][i], strict=strict)
                else:
                    self.model.load_state_dict(ch['state_dict'], strict=strict)

            if self.optimizer is not None and optimizer:
                self.optimizer.load_state_dict(ch['optimizer'])
            print("=> Loaded! Epoch {}".format(ch['epoch']))
        else:
            assert False, "No checkpoint!"

        return (ch['epoch'], ch['best_val_loss'],
                ch['best_val_acc'], ch['waiting_for'])

    def restore_from_singles(self, ckpt='best', model=True):
        ckpt_path = self.best_ckpt.replace(
            'ensembles', 'singles')
        for i in range(len(self.model.models)):
            my_ckpt = ckpt_path.replace('copy__', 'copy_%d' % i)
            ch = torch.load(my_ckpt, map_location=device)
            self.model.models[i].load_state_dict(
                ch['state_dict'], strict=True)

    def restore_from_another_model(self, name, model=False,
                                   ckpt='last', strict=True):
        model_path = os.path.join(PROJECT_ROOT, 'weights', name)

        ckpt = 'checkpoint' if ckpt == 'last' else 'model_best'
        ckpt_path = os.path.join(model_path, '%s.pth.tar' % ckpt)
        ch = torch.load(ckpt_path, map_location=device)

        if model:
            self.model.load_state_dict(ch['state_dict'], strict=strict)

        print('Restored base model %s' % name)

        return (ch['epoch'], ch['best_val_loss'],
                ch['best_val_acc'], ch['waiting_for'])

    def save_checkpoint(self, epoch, best_val_acc, best_val_loss, waiting_for,
                        is_best, filename='checkpoint.pth.tar', optimizer=None,
                        extra=None):
        state = {'epoch': epoch + 1,
                 'args': self.args,
                 'best_val_acc': best_val_acc,
                 'best_val_loss': best_val_loss,
                 'waiting_for': waiting_for,
                }

        if extra is not None:
            state.update(extra)

        if not isinstance(self.model, nn.Module):
            state['state_dict'] = [m.state_dict() for m in self.model.models]
        else:
            state['state_dict'] = self.model.state_dict()

        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        model_path = check_dir(self.model_path, True, False)
        torch.save(state, os.path.join(model_path, filename))
        if is_best:
            shutil.copyfile(os.path.join(model_path, filename),
                            os.path.join(model_path, 'model_best.pth.tar'))
