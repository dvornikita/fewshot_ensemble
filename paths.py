import os
import sys

PROJECT_ROOT = '/'.join(os.path.realpath(__file__).split('/')[:-1])
IMAGENET_ROOT = None    # give a path to the original imagenet
MINI_IMAGENET_ROOT = IMAGENET_ROOT

# In case you want to work with these datasets too...
OMNIGLOT_ROOT = None    # put your path here if you want to use this dataset
CUB_ROOT = None    # put your path here if you want to use this dataset
TIERED_IMAGENET_ROOT = None    # put your path here if you want to use this dataset
COCO_ROOT = None    # put your path here if you want to use this dataset
