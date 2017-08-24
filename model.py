
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def generator_simplified_api(inputs, is_train=True, reuse=False):
    image_size = 64
    gf_dim = 64 # Dimension of gen filters in first conv layer. [64]
    c_dim = FLAGS.c_dim # n_color 3
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        n = InputLayer(inputs, name='g/in')

        n = \
            DenseLayer(
                n,
                n_units=gf_dim*8*8*8,
                W_init=w_init,
                b_init=None,
                act=tf.identity,
                name='g/h0/lin'
            )

        n = \
            ReshapeLayer(
                n,
                shape=[-1, 8, 8, gf_dim*8],
                name='g/h0/reshape',
            )

        n = \
            BatchNormLayer(
                n,
                act=tf.nn.relu,
                is_train=is_train,
                gamma_init=gamma_init,
                name='g/h0/batch_norm'
            )
        n.outputs = tf.depth_to_space(n.outputs, 2)

        n = \
            Conv2d(
                n,
                gf_dim*4,
                (5, 5),
                padding='VALID',
                act=None,
                W_init=w_init,
                b_init=None,
                name='g/h1/decon2d',
            )

        n = \
            BatchNormLayer(
                n,
                act=tf.nn.relu,
                is_train=is_train,
                gamma_init=gamma_init,
                name='g/h1/batch_norm'
            )
        n.outputs = tf.depth_to_space(n.outputs, 2)

        n = \
            Conv2d(
                n,
                gf_dim*2,
                (6, 6),
                padding='VALID',
                act=None,
                W_init=w_init,
                b_init=None,
                name='g/h2/decon2d'
            )

        n = \
            BatchNormLayer(
                n,
                act=tf.nn.relu,
                is_train=is_train,
                gamma_init=gamma_init,
                name='g/h2/batch_norm'
            )
        n.outputs = tf.depth_to_space(n.outputs, 2)

        n = \
            Conv2d(
                n,
                gf_dim,
                (5, 5),
                padding='VALID',
                act=None,
                W_init=w_init,
                b_init=None,
                name='g/h3/decon2d'
            )

        n = \
            BatchNormLayer(
                n,
                act=tf.nn.relu,
                is_train=is_train,
                gamma_init=gamma_init,
                name='g/h3/batch_norm'
            )
        n.outputs = tf.depth_to_space(n.outputs, 2)

        n = \
            Conv2d(
                n,
                c_dim,
                (5, 5),
                padding='VALID',
                act=None,
                W_init=w_init,
                name='g/h4/decon2d'
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
        net_h0.outputs = tf.space_to_depth(net_h0.outputs, 2)

        net_h1 = \
            Conv2d(
                net_h0,
                df_dim*2,
                (5, 5),
                act=None,
                padding='VALID',
                W_init=w_init,
                b_init=None,
                name='d/h1/conv2d'
            )
        net_h1.outputs = tf.space_to_depth(net_h1.outputs, 2)

        net_h1 = \
            BatchNormLayer(
                net_h1,
                act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train,
                gamma_init=gamma_init,
                name='d/h1/batch_norm'
            )

        net_h2 = \
            Conv2d(
                net_h1,
                df_dim*4,
                (3, 3),
                act=None,
                padding='VALID',
                W_init=w_init,
                b_init=None,
                name='d/h2/conv2d',
            )

        net_h2 = \
            BatchNormLayer(
                net_h2,
                act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train,
                gamma_init=gamma_init,
                name='d/h2/batch_norm',
            )

        net_h3 = \
            Conv2d(
                net_h2,
                df_dim*4,
                (4, 4),
                act=None,
                padding='VALID',
                W_init=w_init,
                b_init=None,
                name='d/h2/conv2d',
            )

        net_h3 = \
            BatchNormLayer(
                net_h3,
                act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train,
                gamma_init=gamma_init,
                name='d/h2/batch_norm',
            )

        net_h4 = \
            FlattenLayer(
                net_h3,
                name='d/h4/flatten',
            )

        net_h4 = \
            DenseLayer(
                net_h4,
                n_units=1,
                act=tf.identity,
                W_init=w_init,
                name='d/h4/lin_sigmoid',
            )

        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    return net_h4, logits
