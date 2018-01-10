"""
Test data dependent initialization.

Example usage:
CUDA_VISIBLE_DEVICES="" python test.py --data_dir ~/pxpp/data
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/local_home/tim/pxpp/data', help='Location for the dataset')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
# optimization
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
if args.data_set == 'cifar':
    import data.cifar10_data as cifar10_data
    DataLoader = cifar10_data.DataLoader
elif args.data_set == 'imagenet':
    import data.imagenet_data as imagenet_data
    DataLoader = imagenet_data.DataLoader
else:
    raise("unsupported dataset")
train_data = DataLoader(args.data_dir, 'train', args.batch_size, rng=rng, shuffle=True, return_labels=False)
obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
h_init = None

# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity}
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
init_pass = model(x_init, h_init, init=True, dropout_p=0.0, **model_opt)
init_preactivations = [op.outputs[0] for op in tf.get_default_graph().get_operations()
        if "preactivation" in op.name]
test_pass = model(x_init, h_init, init=False, dropout_p=0.0, **model_opt)
preactivations = [op.outputs[0] for op in tf.get_default_graph().get_operations()
        if "preactivation" in op.name and not op.outputs[0] in init_preactivations]
# compute moments of preactivations
mes = list()
ves = list()
for pa in preactivations:
    r = len(pa.shape.as_list()) - 1
    m, v = tf.nn.moments(pa, list(range(r)))
    me = tf.abs(m - tf.zeros_like(m))
    me = tf.reduce_max(me)
    ve = tf.abs(v - tf.ones_like(v))
    ve = tf.reduce_max(ve)
    mes.append(me)
    ves.append(ve)

# init & save
initializer = tf.global_variables_initializer()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        raise NotImplemented()
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            raise NotImplemented()
    else:
        raise NotImplemented()
    return feed_dict

# //////////// perform training //////////////
with tf.Session() as sess:
    # init
    train_data.reset()  # rewind the iterator back to 0 to do one full epoch
    print('initializing the model...')
    sess.run(initializer)
    feed_dict = make_feed_dict(train_data.next(args.init_batch_size), init=True)  # manually retrieve exactly init_batch_size examples
    print("Max. deviation from target moments before data dependent init")
    info_ms, info_vs = sess.run([mes,ves], feed_dict)
    print(np.max(info_ms), np.max(info_vs))
    print("Running data dependent init")
    sess.run(init_pass, feed_dict)
    print("Max. deviations from target moments after data dependent init:")
    info_ms, info_vs = sess.run([mes,ves], feed_dict)
    print(np.max(info_ms), np.max(info_vs))
