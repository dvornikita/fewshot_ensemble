import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


class AddRandomNoise():
    def __init__(self, sigma, prob, modulate=False):
        self.sigma = sigma
        self.prob = prob
        self.modulate = modulate

    def __call__(self, x):
        noise = torch.randn_like(x) * self.sigma
        if self.modulate:
            noise = noise * torch.rand(1)
        return torch.clamp(x + noise, 0, 1)


class BatchIdxSampler():
    def __init__(self, dataset, batch_size=16, n_copy=1,
                 shuffle=True, random_batch=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_batch = random_batch
        self.n_copy = n_copy

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idxs)

        for i in range(len(self)):
            if self.random_batch:
                batch_idxs = random.sample(idxs, self.batch_size)
            else:
                batch_idxs = idxs[i*self.batch_size: (i+1)*self.batch_size]
            yield batch_idxs * self.n_copy


def get_omniglot_transform(*args, **kwargs):
    def omniglot_to_tensor(img):
        img = np.array(img, dtype=np.float32)[None, :, :]
        return torch.FloatTensor(img)
    return omniglot_to_tensor


def get_mini_imagenet_transform_augmented(backbone, mode, aug_dict):
    train = mode == 'train'
    if train and aug_dict['random_aug']:
        aug_dict = get_random_aug_params()
        print('AUG DICT', aug_dict)

    transform_list = []
    if backbone == 'deep':
        resize, crop_size = None, 224
    elif backbone == 'wide':
        resize, crop_size = 92, 80
    else:
        resize, crop_size = 96, 84

    if resize:
        transform_list.append(transforms.Resize(resize))

    TrimFn = transforms.RandomCrop if train else transforms.CenterCrop
    transform_list.append(TrimFn(crop_size))

    if train:
        transform_list.append(transforms.RandomHorizontalFlip())

    if train and aug_dict['color_jitter']:
        color_params = {k.split('_')[-1]: aug_dict[k] for k in aug_dict
                        if 'color' in k and 'jitter' not in k}
        transform_list.append(transforms.ColorJitter(**color_params))

    transform_list.append(transforms.ToTensor())

    if train and aug_dict['random_noise']:
        transform_list.append(AddRandomNoise(aug_dict['noise_sigma'],
                                             aug_dict['noise_prob'],
                                             aug_dict['noise_modulate']))

    if backbone == 'wide':
        mean, std = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.262])
    else:
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)


def get_cub_transform_augmented(**kwargs):
    train = kwargs['mode'] == 'train'
    transform_list = []
    if kwargs['backbone'] == 'deep':
        resize, crop_size = (228, 228), (224, 224)
    elif kwargs['backbone'] == 'wide':
        resize, crop_size = (84, 84), (80, 80)
    else:
        resize, crop_size = 88, 84

    if not train:
        resize = crop_size

    transform_list.append(transforms.Resize(
        resize, interpolation=Image.BICUBIC))

    if train:
        transform_list.append(transforms.RandomCrop(crop_size))
    else:
        transform_list.append(transforms.CenterCrop(crop_size))

    if train:
        transform_list.append(transforms.RandomHorizontalFlip())

    if train and kwargs['color_jitter']:
        color_params = {'brightness': 0.30, 'saturation': 0.30,
                        'contrast': 0.3, 'hue': 0.03}
        transform_list.append(transforms.ColorJitter(**color_params))

    transform_list.append(transforms.ToTensor())

    if train and kwargs['random_noise']:
        transform_list.append(AddRandomNoise(sigma=0.1))

    if not kwargs.get('debug', False):
        if kwargs['backbone'] == 'wide':
            mean, std = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.262])
        else:
            mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)


def sample_to_device(sample, device='cpu'):
    return {'images': sample['images'].to(device),
            'labels': sample['labels'].to(device),
            'names': sample['names']}


def unnormalize(tarr):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32,
                           device=tarr.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32,
                          device=tarr.device)
    tarr = tarr.mul(std[None, :, None, None]).add(mean[None, :, None, None])
    return tarr


def back_to_image(tarr, pil=True):
    if len(tarr.shape) == 3:
        tarr = tarr.unsqueeze(0)
    tarr = unnormalize(tarr)
    arr = tarr.to('cpu').permute([0, 2, 3, 1])
    print(arr.shape)
    if pil:
        return Image.fromarray((arr[0].numpy() * 255).astype('uint8'))
    else:
        return arr


def get_random_aug_params():
    aug_dict = {}
    aug_dict['resize'] = np.random.randint(238, 257)
    aug_dict['color_jitter'] = bool(np.random.binomial(1, 0.7))
    aug_dict['color_brightness'] = np.random.uniform(0, 0.7)
    aug_dict['color_saturation'] = np.random.uniform(0, 0.7)
    aug_dict['color_contrast'] = np.random.uniform(0, 0.7)
    aug_dict['color_hue'] = np.random.uniform(0, 0.08)
    aug_dict['random_noise'] = bool(np.random.binomial(1, 0.7))
    aug_dict['noise_prob'] = np.random.uniform(0, 0.8)
    aug_dict['noise_sigma'] = np.random.uniform(0, 0.15)
    aug_dict['noise_modulate'] = bool(np.random.binomial(1, 0.4))
    aug_dict['box_out'] = False
    return aug_dict
