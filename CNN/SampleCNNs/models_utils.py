import tensorflow as tf


def mobiconv_fixed_padding(inputs, filters, kernel_size, strides, data_format, layer_name):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)    
    
    depthwise_filter = tf.get_variable(name=layer_name+"depth_filter", shape=(kernel_size, kernel_size, filters, 1), dtype=tf.float32,
                        initializer=tf.variance_scaling_initializer(),
                        trainable=True)
    pointwise_filter = tf.get_variable(name=layer_name+"pointwise_filter", shape=(1, 1, filters, filters*4), dtype=tf.float32,
                        initializer=tf.variance_scaling_initializer(),
                        trainable=True)
    # First get the separable Layers
    return tf.nn.separable_conv2d(inputs,
        depthwise_filter,
        pointwise_filter,
        [1, 1, strides, strides],
        padding=('SAME' if strides == 1 else 'VALID'),
        rate=None,
        data_format="NCHW"
        )

def mobile_bottle_neck(inputs, filters, training, projection_shortcut,
                         strides, use_batchnorm, layer_name, data_format):
    
    shortcut = inputs

    if use_batchnorm:
        norm = batch_norm
    else:
        norm = layer_norm

    inputs = norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    
    # Ectomy
    inputs = mobiconv_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format, layer_name=layer_name)
    inputs = norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    return inputs + shortcut

def mobile_block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, layer_name, use_batchnorm, data_format):

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    use_batchnorm, layer_name + "-l0", data_format)

    for i in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, use_batchnorm, layer_name + "-l" + str(i), data_format)

    return tf.identity(inputs, name)

##############################################################
################### Helper Functions  ########################
##############################################################

def layer_norm(inputs, training, data_format):
    return tf.contrib.layers.layer_norm(
            inputs,
            center=True,
            scale=True,
            activation_fn=tf.nn.relu,
            trainable=True,
            begin_norm_axis= 1 if data_format == 'channels_first' else 3,
            begin_params_axis= 3 if data_format == 'channels_first' else 1,
        )

def classical_head(inputs, name, training, data_format, use_batchnorm, metadata):
    FILTERS, KERNELS, STRIDES, POOL_SIZE, POOL_STRIDE = metadata
    norm = batch_norm if use_batchnorm else layer_norm
    with tf.variable_scope(name):
        inputs = conv2d_fixed_padding(inputs, FILTERS, KERNELS, STRIDES, data_format)

        inputs = norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)
        
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=POOL_SIZE,
            strides=POOL_STRIDE, padding='SAME',
            data_format=data_format)
    return inputs









##############################################################
####### Functions from tensorflow/models/official/resnet  ####
##############################################################


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def bottle_neck(inputs, filters, training, projection_shortcut,
                         strides, use_batchnorm, data_format):
    shortcut = inputs
    if use_batchnorm:
        norm = batch_norm
    else:
        norm = layer_norm
    
    inputs = norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

    return inputs + shortcut
def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
    Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, use_batchnorm, data_format):

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    use_batchnorm, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, use_batchnorm, data_format)

    return tf.identity(inputs, name)

