import tensorflow as tf


def conv2d(input, k_size, input_c, output_c, strides, padding, name):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_size, k_size, input_c, output_c],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input, w, strides=[1, strides, strides, 1], padding= padding)
        biases = tf.get_variable('biases', [output_c], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias