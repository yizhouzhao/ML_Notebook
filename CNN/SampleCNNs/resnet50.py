import tensorflow as tf
from models_utils import *
from utils.cifar10_utils import *
from run_model import per_device_batch_size
import os
FILTERS = 64
KERNELS = 7
STRIDES = 2
POOL_SIZE = 3
POOL_STRIDE = 2

START_FILTER = 64
RESNET_FILTERS = [3, 4, 6, 3]
BLOCK_STRIDES = [2, 2, 2, 2]

FINAL_SIZE = 2048
NUM_CLASSES = 10
DATA_DIR = "data_dir"
BATCH_SIZE = 128
NUM_EPOCHS = 1000

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

def resnet_model(features, training, use_batchnorm, data_format, name="resnet_50"):
    with tf.variable_scope(name):
        inputs = resnet_model_no_last_layer(features, training, use_batchnorm, data_format, name)
        inputs = tf.layers.dense(inputs=inputs, units=NUM_CLASSES)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs

def resnet_model_no_last_layer(features, training, use_batchnorm, data_format, name="resnet_50"):
    with tf.variable_scope(name):
        inputs = tf.transpose(features, (0, 3, 1, 2))
        meta_data = (FILTERS, KERNELS, STRIDES, POOL_SIZE, POOL_STRIDE)
        inputs = classical_head(inputs, name, training, data_format, use_batchnorm, meta_data)
        inputs = tf.nn.relu(inputs)
        
        for i, num_blocks in enumerate(RESNET_FILTERS):
            num_filters = START_FILTER * (2**i)
            inputs = block_layer(inputs, num_filters, True, bottle_neck, num_blocks,
            strides=BLOCK_STRIDES[i], training=training,
            name='block_layer{}'.format(i + 1), data_format=data_format, use_batchnorm=use_batchnorm)
        
        norm = batch_norm if use_batchnorm else layer_norm

        inputs = norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

        # The current top layer has shape
        # `batch_size x pool_size x pool_size x final_size`.
        # ResNet does an Average Pooling layer over pool_size,
        # but that is the same as doing a reduce_mean. We do a reduce_mean
        # here because it performs better than AveragePooling2D.
        axes = [2, 3] if data_format == 'channels_first' else [1, 2]
        inputs = tf.reduce_mean(inputs, axes, keepdims=True)
        inputs = tf.identity(inputs, 'final_reduce_mean')

        inputs = tf.reshape(inputs, [-1, FINAL_SIZE])
        return inputs

def input_fn_train():
    return input_fn(
        is_training=True, data_dir=DATA_DIR,
        batch_size=per_device_batch_size(
            BATCH_SIZE, 1),
        num_epochs=NUM_EPOCHS,
        num_gpus=1)

def input_fn_eval():
    return input_fn(
        is_training=False, data_dir=DATA_DIR,
        batch_size=per_device_batch_size(
            BATCH_SIZE, 1),
        num_epochs=1)
