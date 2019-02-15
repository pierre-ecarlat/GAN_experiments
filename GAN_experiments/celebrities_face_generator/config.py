"""
Configuration script, require easydict (`pip install easydict`).
Can be imported using:
`from config import cfg`
"""

import os
from easydict import EasyDict as edict

__C = edict()
cfg = __C


##############################################
# General options                            #
##############################################
# Dataset to use (sample has 99 images)
__C.DATASET_NAME = 'celeba' # 'celeba_sample'
# Main directory
__C.ROOT_DIR = os.path.dirname(__file__)
# The directory where to find the datasets
__C.DATASET_DIR = os.path.join(__C.ROOT_DIR, 'datasets', __C.DATASET_NAME)

##############################################
# Training options                           #
##############################################
# Number of workers for dataloader
__C.WORKERS = 2
# Batch size during training
__C.BATCH_SIZE = 128
# Number of training epochs
__C.NB_EPOCHS = 5
# Learning rate
__C.LEARNING_RATE = 0.0002
# Beta1 hyperparam for Adam optimizers
__C.BETA1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
__C.NB_GPUS = 0

##############################################
# Images options                             #
##############################################
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
__C.IMAGE_SIZE = 64
# Number of channels (color images = 3, grey = 1)
__C.NUMBER_CHANNELS = 3

##############################################
# GAN options                                #
##############################################
# Size of z latent vector (i.e. size of generator input)
__C.SIZE_G_INPUT = 100
# Size of feature maps in generator
__C.SIZE_G_FEATURES_MAP = 64
# Size of feature maps in discriminator
__C.SIZE_D_FEATURES_MAP = 64