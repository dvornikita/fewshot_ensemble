# Diversity with Cooperation: Ensemble Methods for Few-Shot Classification

Ensembles of deep neural networks combined with nearest centroid classifiers to
solve few-shot classification. 

## Introduction
This repository contains original implementation of the paper 'Diversity with Cooperation: Ensemble Methods for Few-Shot Classification' by Nikita Dvornik, Cordelia Schmid and Julien Mairal.
The paper is available at https://arxiv.org/abs/1903.11341

## Cite

    @inproceedings{dvornik2019diversity,
      title = {Diversity with Cooperation: Ensemble Methods for Few-Shot Classification,
      author = {Dvornik, Nikita and Schmid, Cordelia and Mairal, Julien},
      booktitle = {{IEEE International Conference on Computer Vision (ICCV)}},
      year = {2019}
    }

## Requirements
* Python 3.5
* PyTorch >= 1.2
* TorchVision >= 0.2
* tqdm

## Preparation
Clone this git repository
```sh
git clone https://github.com/dvornikita/fewshot_ensemble.git
```
and install all the requirements listed above.
In the file `path.py`, specify the path to your datasets. For mini-imagenet and tiered-imagenet, you can specify the path to the full ImageNet dataset. Make sure that your ImageNet version contains classes with names in `splits/`.
You are all set.

## Training and testing a single feature extractor
For example, you want to train a WideResNet28 on mini-ImageNet
```sh
python singles/train.py --model.model_name=wideresnet --data.dataset=mini_imagenet --model.backbone=wide 
```
This command will train a network on a classification task, using the training set of mini-ImageNet.
Here, `--model.model_name` is a required argument that specifies the name of the experiment (needed to restore the weights later in testing and to visualize statistics in Tensorboard), `--data.dataset` specifies the name of the dataset you want to train on. The argument `--model.backbone` specifies the feature extractor you use ("deep" is ResNet18 and "wide" stands for WideResNet). 

To test this model, you should run
```sh
python singles/test.py --model.model_name=wideresnet --data.dataset=mini_imagenet --model.backbone=wide --data.test_shot=5
```
This command will generate 1000 5-shot problems, test previously learned feature extractor "wideresnet" on them and report mean accuracy with confidence intervals. To test 1-shot performance, run the command with `--data.test_shot=1`.

## Training and testing ensembles with relations
For example, if you want to train a Robust ensemble of 10 ResNets on mini-ImageNet
```sh
python ensembles/train.py --model.model_name=deep_robust10 --data.dataset=mini_imagenet --model.backbone=deep --ens.num_heads=10 --ens.relation_type=robust
```
This code will train 10 ResNet18 networks with Robust relations. Here, `ens.num_heads` sets the ensemble size, and `--ens.relation_type` specifies the type of relation between the network you want for training the ensemble (the value is in ['independent', 'cooperation', 'diversity', 'robust']).

To test the model, run:
```sh
python ensembles/test.py --model.model_name=deep_robust10 --data.dataset=mini_imagenet --model.backbone=deep --ens.num_heads=10 --data.test_shot=5
```

## Distilling your ensemble and testing it
1. To distill the ensemble of 10 ResNet18, trained in the previously, run:
```sh
python ensembles/distill.py --model.model_name=robust10_distilled --data.dataset=mini_imagenet --model.backbone=deep --ens.num_heads=10 --distil.name=deep_robust10
```
Importantly, the flag `--distil.name` specifies the name of the full ensemble you want to distill into a single network with name "robust10_distilled". You can change parameters of distillation in config.py file.
To test this model, run:
```sh
python singles/test.py --model.model_name=robust10_distilled --data.dataset=mini_imagenet --model.backbone=wide --data.test_shot=5
```
    
__NOTE__: When distilling the model, it is being automatically placed to `singles/` directory. This is because it if just a single neural network.
