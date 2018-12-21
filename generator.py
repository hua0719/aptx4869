import tensorflow as tf
from tensorflow.contrib import slim
from ops import *


# class Generator(object, input):
#     def __init__(self):
#         self.H, self.W = 112, 80 # Width/Height for ISLES2015 dataset
#         self.input = input

def Upsampling(inputs, scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])


def RCU(input, name):
    c = input.get_shape().as_list()[3]
    conv1 = conv2d(input, k_size=3, input_c=c, output_c=c, strides=1, padding="SAME", name=name+'_conv1')
    act1 = tf.nn.leaky_relu(conv1, name=name+'_act1')

    conv2 = conv2d(act1, k_size=3, input_c=c, output_c=c, strides=1, padding="SAME", name=name+'_conv2')
    act2 = tf.nn.leaky_relu(conv2, name=name + '_act2')
    return  tf.add(input, act2)


def MultiResolutionFusion(high_inputs=None, low_inputs=None):
    n_filters = low_inputs.get_shape().as_list()[3]
    if high_inputs is None: # RefineNet block 4

        fuse = slim.conv2d(low_inputs, n_filters, 3, activation_fn=None)

        return fuse

    else:

        conv_low = slim.conv2d(low_inputs, n_filters, 3, activation_fn=None)
        conv_high = slim.conv2d(high_inputs, n_filters, 3, activation_fn=None)

        conv_low_up = Upsampling(conv_low, 2)

        return tf.add(conv_low_up, conv_high)


def ChainedResidualPooling(inputs):
    """
    Chained residual pooling aims to capture background
    context from a large image region. This component is
    built as a chain of 2 pooling blocks, each consisting
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are
    fused together with the input feature map through summation
    of residual connections.
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
    Returns:
      Double-pooled feature maps
    """
    n_filters = inputs.get_shape().as_list()[3]
    net_relu = tf.nn.relu(inputs)
    net = slim.max_pool2d(net_relu, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(net, n_filters, 3, activation_fn=None)
    net_sum_1 = tf.add(net, net_relu)

    net = slim.max_pool2d(net, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(net, n_filters, 3, activation_fn=None)
    net_sum_2 = tf.add(net, net_sum_1)

    return net_sum_2


def refinenet(high_inputs=None,low_inputs=None, name=None):
    if low_inputs is None: # block 4
        rcu_new_low = RCU(high_inputs, name=name+'rcu1')
        rcu_new_low = RCU(rcu_new_low, name=name+ 'rcu2')

        fuse = MultiResolutionFusion(high_inputs=None, low_inputs=rcu_new_low)
        fuse_pooling = ChainedResidualPooling(fuse)
        output = RCU(fuse_pooling, name=name+'rcu3')
        return output
    else:
        rcu_high = RCU(high_inputs, name=name+'rcu4')
        rcu_high = RCU(rcu_high, name=name+'rcu5')

        fuse = MultiResolutionFusion(rcu_high, low_inputs)
        fuse_pooling = ChainedResidualPooling(fuse)
        output = RCU(fuse_pooling, name=name+'rcu6')
        return output


def encoder(input, modality, is_training = True, reuse=False):
    intput_c = input.get_shape().as_list()[3]
    with tf.variable_scope("generator_encoder" + modality, reuse=reuse):
        #
        # # 卷积核1[3, 3, 1, 32]
        # filter1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
        # # 卷积核2[3, 3, 32, 32]
        # filter2 = tf.Variable(tf.random_normal([3, 3, 32, 32]))

        # block 1
        conv1 = bn(conv2d(input, k_size=3, input_c=intput_c, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b1_conv1'),
                   is_training=is_training,  scope='g_enc_bn1')
        act1 = tf.nn.leaky_relu(conv1, name='g_enc_' + 'block1_act1')
        conv2 = bn(conv2d(act1, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b1_conv2'),
                   is_training=is_training, scope='g_enc_bn2')
        act2 = tf.nn.leaky_relu(conv2, name='g_enc_' + 'block1_act2')

        pool1 = tf.nn.max_pool(act2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding="VALID", name='g_enc_' + 'pool1')

        # block 2
        conv3 = bn(conv2d(pool1, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b2_conv1'),
                   is_training=is_training, scope='g_enc_bn3')
        act3 = tf.nn.leaky_relu(conv3, name='g_enc_' + 'block2_act1')
        conv4 = bn(conv2d(act3, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b2_conv2'),
                   is_training=is_training, scope='g_enc_bn4')
        act4 = tf.nn.leaky_relu(conv4, name='g_enc_' + 'block2_act2')


        pool2 = tf.nn.max_pool(act4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name='g_enc_' + 'pool2')

        # block 3
        conv5 = bn(conv2d(pool2, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b3_conv1'),
                   is_training=is_training, scope='g_enc_bn5')
        act5 = tf.nn.leaky_relu(conv5, name='g_enc_' + 'block3_act1')
        conv6 = bn(conv2d(act5, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b3_conv2'),
                   is_training=is_training, scope='g_enc_bn6')
        act6 = tf.nn.leaky_relu(conv6, name='g_enc_' + 'block3_act2')

        refi1 = refinenet(high_inputs=act6, low_inputs=None, name='refi1')
        refi2 = refinenet(high_inputs=act4, low_inputs=refi1, name='refi2')
        refi3 = refinenet(high_inputs=act2, low_inputs=refi2, name='refi3')

        conv7= bn(conv2d(refi3, k_size=3, input_c=32, output_c=16, strides=1, padding="SAME", name='g_enc_' + 'conv7'),
                    is_training=is_training, scope='g_enc_bn7')
        act7 = tf.nn.leaky_relu(conv7, name='g_enc_' + 'act7')
        return act7





"""TMI原论文编码器"""
def ORI_encoder(input, modality, is_training = True, reuse=False):
    intput_c = input.get_shape().as_list()[3]
    with tf.variable_scope("generator_encoder" + modality, reuse=reuse):
        #
        # # 卷积核1[3, 3, 1, 32]
        # filter1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
        # # 卷积核2[3, 3, 32, 32]
        # filter2 = tf.Variable(tf.random_normal([3, 3, 32, 32]))

        # block 1
        conv1 = bn(conv2d(input, k_size=3, input_c=intput_c, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b1_conv1'),
                   is_training=is_training,  scope='g_enc_bn1')
        act1 = tf.nn.leaky_relu(conv1, name='g_enc_' + 'block1_act1')
        conv2 = bn(conv2d(act1, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b1_conv2'),
                   is_training=is_training, scope='g_enc_bn2')
        act2 = tf.nn.leaky_relu(conv2, name='g_enc_' + 'block1_act2')

        pool1 = tf.nn.max_pool(act2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding="VALID", name='g_enc_' + 'pool1')

        # block 2
        conv3 = bn(conv2d(pool1, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b2_conv1'),
                   is_training=is_training, scope='g_enc_bn3')
        act3 = tf.nn.leaky_relu(conv3, name='g_enc_' + 'block2_act1')
        conv4 = bn(conv2d(act3, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b2_conv2'),
                   is_training=is_training, scope='g_enc_bn4')
        act4 = tf.nn.leaky_relu(conv4, name='g_enc_' + 'block2_act2')

        pool2 = tf.nn.max_pool(act4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name='g_enc_' + 'pool2')

        # block 3
        conv5 = bn(conv2d(pool2, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b3_conv1'),
                   is_training=is_training, scope='g_enc_bn5')
        act5 = tf.nn.leaky_relu(conv5, name='g_enc_' + 'block3_act1')
        conv6 = bn(conv2d(act5, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b3_conv2'),
                   is_training=is_training, scope='g_enc_bn6')
        act6 = tf.nn.leaky_relu(conv6, name='g_enc_' + 'block3_act2')

        ups1 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(act6)
        # 卷积核3
        filter3 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
        conv7 = bn(conv2d(ups1, k_size=3, input_c=32, output_c=64, strides=1, padding="SAME", name='g_enc_' + 'conv7'),
                   is_training=is_training, scope='g_enc_bn7')

        skip1 = tf.concat([act4, conv7], axis=3)

        # block 4
        # 卷积核4 [3, 3, 64, 32]
        filter4 = tf.Variable(tf.random_normal([3, 3, 96, 32]))
        conv8 = bn(conv2d(skip1,  k_size=3, input_c=96, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b4_conv1'),
                   is_training=is_training, scope='g_enc_bn8')
        act7 = tf.nn.leaky_relu(conv8, name='g_enc_' + 'block3_act1')
        # 卷积核5
        filter5 = tf.Variable(tf.random_normal([3, 3, 32, 32]))
        conv9 = bn(conv2d(act7, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b4_conv2'),
                   is_training=is_training, scope='g_enc_bn9')
        act8 = tf.nn.leaky_relu(conv9, name='g_enc_' + 'block3_act2')

        ups2 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(act8)
        # 卷积核6
        filter6 = tf.Variable(tf.random_normal([3, 3, 32, 32]))
        conv10 = bn(conv2d(ups2, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'conv10'),
                    is_training=is_training, scope='g_enc_bn10')
        skip2 = tf.concat([act2, conv10], axis=3)

        # block 5
        # 卷积核7
        filter7 = tf.Variable(tf.random_normal([3, 3, 64, 32]))
        conv11 = bn(conv2d(skip2, k_size=3, input_c=64, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b5_conv1'),
                    is_training=is_training, scope='g_enc_bn11')
        act9 = tf.nn.leaky_relu(conv11, name='g_block5_act1')
        conv12 = bn(conv2d(act9, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_enc_' + 'b5_conv2'),
                    is_training=is_training, scope='g_enc_bn12')
        act10 = tf.nn.leaky_relu(conv12, name='g_enc_' + 'block5_act2')

        # 卷积核8
        filter8 = tf.Variable(tf.random_normal([3, 3, 32, 16]))
        conv13 = bn(conv2d(act10, k_size=3, input_c=32, output_c=16, strides=1, padding="SAME", name='g_enc_' + 'conv13'),
                    is_training=is_training, scope='g_enc_bn13')
        act11 = tf.nn.leaky_relu(conv13, name='g_enc_' + 'act11')           # L*H*W

        # flat = tf.reshape(act11, [batch_size*2, -1])      # bs2*N
        # latent = bn(linear(flat, 1024, scope='g_en_fc1'), is_training=is_training, scope='g_en_bn1')   #bs2*1024
        # act12 = tf.nn.leaky_relu(latent, name='enc_' + 'act12')
        #
        # gaussian_params = linear(act12, 2 * z_dim, scope='g_en_fc2')         #bs2*z2
        #
        # mean = gaussian_params[:, :z_dim]
        # stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, z_dim:])

        return act11                               # , mean, stddev




def decoder(input,is_training=True, reuse=False):

    with tf.variable_scope("decoder", reuse=reuse):
        # block 1
        # 卷积核1[3, 3, 16, 32]
        # filter1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
        # 卷积核2[3, 3, 32, 32]
        # filter2 = tf.Variable(tf.random_normal([3, 3, 32, 32]))
        conv1 = bn(conv2d(input, k_size=3, input_c=16, output_c=32, strides=1, padding="SAME", name='g_dec_' + 'b1_conv1'),
                   is_training=is_training, scope='g_dec_bn1')
        act1 = tf.nn.relu(conv1)
        conv2 = bn(conv2d(act1, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_dec_' + 'b1_conv2'),
                   is_training=is_training, scope='g_dec_bn2')
        skip1 = tf.concat([input, conv2], axis=3)

        # block 2
        conv3 = bn(conv2d(skip1, k_size=3, input_c=48, output_c=32, strides=1, padding="SAME", name='g_dec_' + 'b2_conv1'),
                   is_training=is_training, scope='g_dec_bn3')
        act3 = tf.nn.relu(conv3)
        conv4 = bn(conv2d(act3, k_size=3, input_c=32, output_c=32, strides=1, padding="SAME", name='g_dec_' + 'b2_conv2'),
                   is_training=is_training, scope='g_dec_bn4')
        act4 = tf.nn.relu(conv4)
        skip2 = tf.concat([skip1, act4], axis=3)

        # 卷积核3[1, 1, 32, 1]
        # filter3 = tf.Variable(tf.random_normal([1, 1, 32, 1]))
        conv5 = conv2d(skip2, k_size=1, input_c=80, output_c=1, strides=1, padding="SAME", name='g_dec_' + 'conv5')
        act5 = tf.nn.relu(conv5)

        return act5
