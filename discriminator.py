import tensorflow as tf
from ops import *


batch_size = 4

# class Discriminator(object, input):
#     def __init__(self):
#         self.H, self.W = 112, 80  # Width/Height for ISLES2015 dataset
#         self.input = input




def strides_d_model( x, is_training=True, reuse=False):                # 来源 超分GAN
    batch_size = x.shape()[0]
    with tf.variable_scope("discriminator", reuse=reuse):

        conv1 = conv2d(input=x, k_size=3, input_c=1, output_c=32, strides=1, padding='SAME', name='d_conv1')
        act1 = tf.nn.leaky_relu(conv1)

        conv2 = bn(conv2d(act1, k_size=3, input_c=32, output_c=32, strides=2, padding='SAME', name='d_conv2'),
                   is_training=is_training, scope='d_bn1')
        act2 = tf.nn.leaky_relu(conv2)

        conv3 = bn(conv2d(act2, k_size=3, input_c=32, output_c=64, strides=1, padding='SAME', name='d_conv3'),
                   is_training=is_training, scope='d_bn2')
        act3 = tf.nn.leaky_relu(conv3)

        conv4 = bn(conv2d(act3, k_size=3, input_c=64, output_c=64, strides=2, padding='SAME', name='d_conv4'),
                   is_training=is_training, scope='d_bn3')
        act4 = tf.nn.leaky_relu(conv4)

        conv5 = bn(conv2d(act4, k_size=3, input_c=64, output_c=128, strides=1, padding='SAME', name='d_conv5'),
                   is_training=is_training, scope='d_bn4')
        act5 = tf.nn.leaky_relu(conv5)

        conv6 = bn(conv2d(act5, k_size=3, input_c=128, output_c=128, strides=2, padding='SAME', name='d_conv6'),
                   is_training=is_training, scope='d_bn5')
        act6 = tf.nn.leaky_relu(conv6)

        flat = tf.reshape(act6, [batch_size*2, -1])            # bs*2,N
        fc1 = linear(flat, 1024, scope='d_fc1')    # bs*2,124
        act7 = tf.nn.leaky_relu(fc1)

        out_logit = linear(act7, 1, scope='d_fc2')        # bs*2, 1

        out = tf.nn.sigmoid(out_logit)

        return out, out_logit, act7


def d_model(x, is_training=True, reuse=False):                # 来源 SegAN
    batch_size = x.get_shape().as_list()[0]
    with tf.variable_scope("discriminator", reuse=reuse):

        conv1 = conv2d(input=x, k_size=7, input_c=1, output_c=32, strides=2, padding='SAME', name='d_conv1')
        act1 = tf.nn.leaky_relu(conv1)                                  # 112*80*32

        conv2 = bn(conv2d(act1, k_size=5, input_c=32, output_c=64, strides=2, padding='SAME', name='d_conv2'),
                   is_training=is_training, scope='d_bn1')              # 56*40*64
        act2 = tf.nn.leaky_relu(conv2)

        conv3 = bn(conv2d(act2, k_size=4, input_c=64, output_c=128, strides=2, padding='SAME', name='d_conv3'),
                   is_training=is_training, scope='d_bn2')              # 28*20*128
        act3 = tf.nn.leaky_relu(conv3)

        conv4 = bn(conv2d(act3, k_size=4, input_c=128, output_c=256, strides=2, padding='SAME', name='d_conv4'),
                   is_training=is_training, scope='d_bn3')              # 14*10*256
        act4 = tf.nn.leaky_relu(conv4)

        conv5 = bn(conv2d(act4, k_size=3, input_c=256, output_c=512, strides=2, padding='SAME', name='d_conv5'),
                   is_training=is_training, scope='d_bn4')               # 7*5*512
        act5 = tf.nn.leaky_relu(conv5)

        flat = tf.reshape(act5, [batch_size, -1])            # bs,N
        fc1 = linear(flat, 1024, scope='d_fc1')    # bs,124
        act6 = tf.nn.leaky_relu(fc1)

        out_logit = linear(act6, 1, scope='d_fc2')        # bs, 1

        out = tf.nn.sigmoid(out_logit)

        # segAN
        scale_out = tf.concat([1 * tf.reshape(x, [batch_size, -1]), 1 * tf.reshape(act1, [batch_size, -1]),
                             2 * tf.reshape(act2, [batch_size, -1]), 2 * tf.reshape(act3, [batch_size, -1]),
                             2 * tf.reshape(act4, [batch_size, -1]), 4 * tf.reshape(act5, [batch_size, -1])], axis=1)

        return out, out_logit, act6, scale_out




