import argparse

parser = argparse.ArgumentParser(description='Train prototypical networks')

# TODO clean the flags

# data args
parser.add_argument('--data.dataset', type=str, default='mini_imagenet', metavar='DS',
                    help="data set name (default: {:s})".format('imageet'))
parser.add_argument('--data.way', type=int, default=60, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=5, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.trainval', action='store_true', help="run in train+validation mode (default: False)")
parser.add_argument('--data.batch_size', type=int, default=16, metavar='NEPOCHS',
                    help='number of workers that pre-process images in parallel')
parser.add_argument('--data.num_workers', type=int, default=16, metavar='NEPOCHS',
                    help='number of workers that pre-process images in parallel')
parser.add_argument('--data.skip_loc', type=int, default=-1, metavar='SKIP_LOC',
                    help='For normal datasets, from wchich index in a cat start skipping samples')
parser.add_argument('--data.skip_size', type=int, default=100, metavar='SKIP_SIZE',
                    help='How many samples to skip')
parser.add_argument('--data.valset', type=str, default='test', metavar='DS',
                    help="data set for validation".format('imageet'))
parser.add_argument('--data.lmdb', action='store_true', help="If Datasets are in lmdb format")

# model args
default_model_name = 'protonet'
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default=3, metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=64, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")
parser.add_argument('--model.backbone', choices=['deep', 'deep_small', 'wide', 'shallow'], default='deep',
                    help="Use ResNet18 for experiments (default: False)")
parser.add_argument('--model.classifier', type=str, default='linear', choices=['', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")
parser.add_argument('--model.sgd_test_steps', type=int, default=100, metavar='STEPS',
                    help="if running test_sgd, running test optimization for this number of steps")
parser.add_argument('--model.dataset', type=str, default='', metavar='MODEDATA',
                    help="dataset the model was trained on")

# ansamble args
parser.add_argument('--ens.num_heads', type=int, default=10, metavar='N_HEADS',
                    help="dimensionality of input images (default: 64)")
parser.add_argument('--ens.head_prob', type=float, default=1, metavar='HEAD_PROB',
                    help="dimensionality of input images (default: 64)")
parser.add_argument('--ens.head_dropout', type=float, default=0.0, metavar='HEAD_DROPOUT',
                    help="dimensionality of input images (default: 64)")

parser.add_argument('--ens.relation_type', type=str, default='independent',
                    choices=['cooperation', 'diversity', 'robust', 'independent'],
                    help="Type of relation ensemble members must have for training")
parser.add_argument('--ens.joint_training', action='store_true', help="Different heads recieve different classes")
parser.add_argument('--ens.rel_fn', type=str, default='cos', choices=['cos', 'neg_cos', 'l2', 'neg_l2', 'js', 'neg_js', 'distil'],
                    help="Relation loss function")
parser.add_argument('--ens.rel_T', type=float, default=1, metavar='DIV_T',
                    help="softmax temperature in relation loss")
parser.add_argument('--ens.rel_coef', type=float, default=0.01,
                    help="Multiplication coefficient for relation loss")
parser.add_argument('--ens.rel_active_epoch', type=float, default=0, metavar='DIV_T',
                    help="softmax temperature in relation loss")
parser.add_argument('--ens.robust_matching', action='store_true',
                    help="Single image augmented in different ways goes to different nets, then the modified softmax outputs are matched")


# Distillation flags
parser.add_argument('--distil.name', type=str, default='',
                    help="The name of an ansamble we distill from")
parser.add_argument('--distil.backbone', choices=['deep', 'deep_small', 'wide', 'shallow'], default='',
                    help="Use ResNet18 for experiments (default: False)")
parser.add_argument('--distil.T', type=float, default=5, help="distilation temperature")
parser.add_argument('--distil.gamma', type=float, default=0.8, help="percentage of distil loss contribution")
parser.add_argument('--distil.unsup_set', type=str, default='val', help="percentage of distil loss contribution")
parser.add_argument('--distil.unsup_size', type=int, default=0, help="distilation unsup size")


# train args
parser.add_argument('--train.determ', action='store_true', help="Initializing with fixed seeds")
parser.add_argument('--train.set', type=str, default='train',
                    metavar='NCOPY', help='The number of test episodes sampled')
parser.add_argument('--train.epochs', type=int, default=300, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optimizer', type=str, default='adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=15, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
parser.add_argument('--train.decay_coef', type=int, default=10, metavar='DECAYCOEF',
                    help='the value to divide learning rate by when decayin lr')
default_weight_decay = 5e-4
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))
parser.add_argument('--train.ckpt_freq', type=int, default=10, metavar='CKPT',
                    help='Saving model params after each CKPT epochs (default: 1000)')
parser.add_argument('--train.resume', type=int, default=1, help="Resume training starting from the last checkpoint (default: True)")
parser.add_argument('--train.patience', type=int, default=25, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')
parser.add_argument('--train.random_batch', action='store_true', help="To construct consecutive tasks, the classes are drawn with replacement (default: False)")
parser.add_argument('--train.spectral_reg', action='store_true', help="Training with spectral reg")
parser.add_argument('--train.spectral_reg_coef', type=float, default=0.01, help="Training with spectral reg")
parser.add_argument('--train.filter_cats', action='store_true', help="Different heads recieve different classes")
parser.add_argument('--train.wide_dropout', type=float, default=0.5, help="Adding dropout inside a basic block of widenet")


# data augmentation args
parser.add_argument('--aug.random_aug', action='store_true', help="Generates Random parameters for data augmentation")
parser.add_argument('--aug.resize', type=int, default=0, help="")
#colors params
parser.add_argument('--aug.color_jitter', action='store_true', help="Do random color jittering as data augmentation")
parser.add_argument('--aug.color_brightness', type=float, default=0.3, help="")
parser.add_argument('--aug.color_saturation', type=float, default=0.3, help="")
parser.add_argument('--aug.color_contrast', type=float, default=0.3, help="")
parser.add_argument('--aug.color_hue', type=float, default=0.03, help="")
#noise params
parser.add_argument('--aug.random_noise', action='store_true', help="To add random noise as data augmentation")
parser.add_argument('--aug.noise_prob', type=float, default=0.9, help="")
parser.add_argument('--aug.noise_sigma', type=float, default=0.01, help="")
parser.add_argument('--aug.noise_modulate', type=bool, default=1, help="")
#box_out params


# test args
parser.add_argument('--test.set', type=str, default='test', choices=['train', 'test', 'val', 'test_new'],
                    metavar='NCOPY', help='The number of test episodes sampled')
parser.add_argument('--test.size', type=int, default=1000, metavar='NCOPY',
                    help='The number of test episodes sampled')
parser.add_argument('--test.seed', type=int, default=1234, metavar='NCOPY',
                    help='seed for reproduc')
parser.add_argument('--test.determ', action='store_true', help="Determenistic test scenario with random seeds fixed")
parser.add_argument('--test.n_copy', type=int, default=1, metavar='NCOPY',
                    help='Number of copies of an image to make for augmentation')
parser.add_argument('--test.augment_query', action='store_true', help="If to augment query images in order to avearge the embeddings")
parser.add_argument('--test.augment_support', action='store_true', help="If to augment support images in order to avearge the embeddings")
parser.add_argument('--test.distance', type=str, choices=['cos', 'l2'], default='cos', help="If to augment support images in order to avearge the embeddings")
parser.add_argument('--test.pca_dim', type=int, help="If > 0, Apply PCA of dim to embeddings at test time")
parser.add_argument('--test.label_prop', action='store_true', help="If to augment support images in order to avearge the embeddings")
parser.add_argument('--test.noprint', action='store_true', help="Doesn't drop the accuracy into file")

# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

args = vars(parser.parse_args())

if args['model.dataset'] == '':
    args['model.dataset'] = args['data.dataset']

if args['distil.backbone'] == '':
    args['distil.backbone'] = args['model.backbone']

if args['ens.relation_type'] != 'independent':
    args['ens.join_training'] = True
    if args['ens.relation_type'] == 'cooperation' or args['ens.relation_type'] == 'robust':
        args['ens.rel_fn'] = 'js'
        args['ens.rel_T'] = 1
        args['ens.rel_coef'] = 1e1

    if args['ens.relation_type'] == 'diversity':
        args['ens.rel_fn'] = 'cos'
        args['ens.rel_T'] = 5
        args['ens.rel_coef'] = 1e0

    if args['ens.relation_type'] == 'robust':
        args['ens.robust_matching'] = True
        args['ens.head_prob'] = 0.7
        args['ens.head_dropout'] = 0.1
