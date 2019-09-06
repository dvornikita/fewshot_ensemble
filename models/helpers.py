import torch
from functools import partial

from models.resnets import resnet18
from models.wide_resnet import wide_resnet
from models.ensembles import Ensemble
from utils import device


def get_model(num_classes, args, ensemble=False):
    if 'deep' in args['model.backbone']:
        model_fn = resnet18
    elif args['model.backbone'] == 'wide':
        model_fn = partial(wide_resnet, dropout=args['train.wide_dropout'])

    if ensemble:
        model = Ensemble(model_fn, num_heads=args['ens.num_heads'],
                         num_classes=num_classes,
                         prob=args['ens.head_prob'],
                         dropout=args['ens.head_dropout'])
    else:
        model = model_fn(classifier=args['model.classifier'],
                         num_classes=num_classes)
    model.to(device)
    return model


def get_optimizer(model, args):
    if args['train.optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args['train.learning_rate'],
                                     weight_decay=args['train.weight_decay'])
    elif args['train.optimizer'] == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args['train.learning_rate'],
                                    momentum=0.9, nesterov=True,
                                    weight_decay=args['train.weight_decay'])
    else:
        assert False, 'No such optimizer'
    return optimizer
