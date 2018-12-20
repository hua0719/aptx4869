import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import *




# class Discriminator(object, input):
#     def __init__(self):
#         self.H, self.W = 112, 80  # Width/Height for ISLES2015 dataset
#         self.input = input

# def attention_block(g, x, f_in, conv_bn):      # attention block
#     f_out = f_in/2
#     W_g = conv_bn(g, f_out, 1, 1)
#     W_x = conv_bn(x, f_out, 1, 1)
#
#     W_gx = relu(W_g + W_x)
#
#     psi = conv_bn(W_gx, 1, 1, 1)
#     psi = tf.nn.sigmoid(psi)
#     return psi*x

def attention_block_d(g, x, f_in, name, is_training, scope):      # discriminator attention block
    f_out = f_in/2
    # W_g = conv_bn(g, f_out, 1, 1)
    W_g = bn(conv2d(g, k_size=1, input_c=f_in, output_c=f_out, strides=1, padding='SAME', name=name+'_1'),
               is_training=is_training, scope=scope+'_1')
    # W_x = conv_bn(x, f_out, 5, 2)
    x_shape = x.get_shape().as_list()[3]
    W_x = bn(conv2d(x, k_size=5, input_c=x_shape, output_c=f_out, strides=2, padding='SAME', name=name+'_2'),
             is_training=is_training, scope=scope+'_2')
    W_gx = tf.nn.relu(W_g + W_x)

    # psi = conv_bn(W_gx, 1, 1, 1)
    psi = bn(conv2d(W_gx, k_size=1, input_c=f_out, output_c=1, strides=1, padding='SAME', name=name+'_3'),
             is_training=is_training, scope=scope+'_3')
    psi = tf.nn.sigmoid(psi)
    return psi*g



def strides_d_model( x, is_training=True, reuse=False):        # 来源 超分GAN
    batch_size = x.get_shape().as_list()[0]
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

        atte1 = attention_block_d(act1, x, f_in=32, name='atte1', is_training=is_training, scope='a_bn1')

        conv2 = bn(conv2d(act1, k_size=5, input_c=32, output_c=64, strides=2, padding='SAME', name='d_conv2'),
                   is_training=is_training, scope='d_bn1')              # 56*40*64
        act2 = tf.nn.leaky_relu(conv2)

        atte2 = attention_block_d(act2, act1, f_in=64, name='atte2', is_training=is_training, scope='a_bn2')

        conv3 = bn(conv2d(act2, k_size=4, input_c=64, output_c=128, strides=2, padding='SAME', name='d_conv3'),
                   is_training=is_training, scope='d_bn2')              # 28*20*128
        act3 = tf.nn.leaky_relu(conv3)

        atte3 = attention_block_d(act3, act2, f_in=128, name='atte3', is_training=is_training, scope='a_bn3')

        conv4 = bn(conv2d(act3, k_size=4, input_c=128, output_c=256, strides=2, padding='SAME', name='d_conv4'),
                   is_training=is_training, scope='d_bn3')              # 14*10*256
        act4 = tf.nn.leaky_relu(conv4)

        atte4 = attention_block_d(act4, act3, f_in=256, name='atte4', is_training=is_training, scope='a_bn4')

        conv5 = bn(conv2d(act4, k_size=3, input_c=256, output_c=512, strides=2, padding='SAME', name='d_conv5'),
                   is_training=is_training, scope='d_bn4')               # 7*5*512
        act5 = tf.nn.leaky_relu(conv5)

        atte5 = attention_block_d(act5, act4, f_in=512, name='atte5', is_training=is_training, scope='a_bn5')

        flat = tf.reshape(act5, [batch_size, -1])            # bs,N
        fc1 = linear(flat, 1024, scope='d_fc1')    # bs,124
        act6 = tf.nn.leaky_relu(fc1)

        out_logit = linear(act6, 1, scope='d_fc2')        # bs, 1

        out = tf.nn.sigmoid(out_logit)

        # segAN
        out0 = tf.reshape(x, [batch_size, -1])
        out1 = tf.reshape(atte1, [batch_size, -1])
        out2 = tf.reshape(atte2, [batch_size, -1])
        out3 = tf.reshape(atte3, [batch_size, -1])
        out4 = tf.reshape(atte4, [batch_size, -1])
        out5 = tf.reshape(atte5, [batch_size, -1])

        scale_out = tf.concat([out0, out1, out2, out3, out4, out5], 1)

        return out, out_logit, act6, scale_out




