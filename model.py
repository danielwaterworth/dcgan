
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def resnet_module(input, n, name):
    net = \
        Conv2d(
            input,
            n,
            (3, 3),
            padding='VALID',
            act=None,
            W_init=w_init,
            b_init=None,
            name=name+'/r0/decon2d',
        )

    net = \
        BatchNormLayer(
            net,
            act=tf.nn.relu,
            is_train=is_train,
            gamma_init=gamma_init,
            name=name+'/r0/batch_norm'
        )

    net = \
        Conv2d(
            net,
            n,
            (3, 3),
            padding='VALID',
            act=None,
            W_init=w_init,
            b_init=None,
            name=name+'/r1/decon2d',
        )

    net = \
        BatchNormLayer(
            net,
            act=tf.nn.relu,
            is_train=is_train,
            gamma_init=gamma_init,
            name=name+'/r1/batch_norm'
        )

    input = \
        LambdaLayer(
            input,
            fn=lambda x: x[:, 2:-2, 2:-2, :],
        )

    return ElementwiseLayer([net, input], 'ADD')

def generator_simplified_api(inputs, is_train=True, reuse=False):
    image_size = 64
    gf_dim = 32 # Dimension of gen filters in first conv layer. [32]
    c_dim = FLAGS.c_dim # n_color 3
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        n = InputLayer(inputs, name='g/in')

        n = \
            DenseLayer(
                n,
                n_units=gf_dim*4*8*8,
                W_init=w_init,
                b_init=None,
                act=tf.identity,
                name='g/h0/lin'
            )

        n = \
            BatchNormLayer(
                n,
                act=tf.nn.relu,
                is_train=is_train,
                gamma_init=gamma_init,
                name='g/h0/batch_norm'
            )

        n = \
            ReshapeLayer(
                n,
                shape=[-1, 12, 12, gf_dim*4],
                name='g/h0/reshape',
            )

        n = resnet_module(n, gf_dim*8, 'g/h1')
        n = LambdaLayer(n, lambda x: tf.depth_to_space(x, 2))
        n = resnet_module(n, gf_dim*4, 'g/h2')
        n = LambdaLayer(n, lambda x: tf.depth_to_space(x, 2))
        n = resnet_module(n, gf_dim*2, 'g/h3')
        n = LambdaLayer(n, lambda x: tf.depth_to_space(x, 2))
        n = resnet_module(n, gf_dim, 'g/h4')
        n = LambdaLayer(n, lambda x: tf.depth_to_space(x, 2))

        n = \
            Conv2d(
                n,
                gf_dim,
                (5, 5),
                padding='VALID',
                act=None,
                W_init=w_init,
                name='g/h5/decon2d'
            )

        n = \
            Conv2d(
                n,
                3,
                (5, 5),
                padding='VALID',
                act=None,
                W_init=w_init,
                name='g/h6/decon2d'
            )

        logits = n.outputs
        n.outputs = tf.nn.tanh(n.outputs)
    return n, logits

def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    batch_size = FLAGS.batch_size # 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='d/in')

        net_h0 = \
            Conv2d(
                net_in,
                df_dim,
                (5, 5),
                act=lambda x: tl.act.lrelu(x, 0.2),
                padding='VALID',
                W_init=w_init,
                name='d/h0/conv2d',
            )

        net_h1 = resnet_module(net_h0, df_dim*2, 'd/h1')
        net_h1 = LambdaLayer(net_h1, lambda x: tf.space_to_depth(x, 2))
        net_h2 = resnet_module(net_h1, df_dim*2, 'd/h2')
        net_h2 = LambdaLayer(net_h2, lambda x: tf.space_to_depth(x, 2))
        net_h3 = resnet_module(net_h2, df_dim*2, 'd/h3')
        net_h3 = LambdaLayer(net_h3, lambda x: tf.space_to_depth(x, 2))
        net_h4 = resnet_module(net_h3, df_dim*2, 'd/h4')
        net_h4 = LambdaLayer(net_h4, lambda x: tf.space_to_depth(x, 2))

        net_h5 = \
            FlattenLayer(
                net_h4,
                name='d/h5/flatten',
            )

        net_h5 = \
            DenseLayer(
                net_h5,
                n_units=1,
                act=tf.identity,
                W_init=w_init,
                name='d/h5/lin_sigmoid',
            )

        logits = net_h5.outputs
        net_h5.outputs = tf.nn.sigmoid(net_h5.outputs)
    return net_h5, logits
