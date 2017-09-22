import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses as loss
import math


class resnet(object):
    def __init__(self, input, num_layers):
        if num_layers not in [20, 32, 50]:
            raise Exception('layers should be 20, 32 or 50')
        self.num_layers = num_layers
        if num_layers == 20:
            n = 3
        elif num_layers == 32:
            n = 5
        elif num_layers == 50:
            n = 8
        self._build_net(input, n)




    def _block(self, inputs, num_outputs, weight_decay, scope, down_sample=False):
        with tf.variable_scope(scope):
            num_inputs = inputs.get_shape().as_list()[3]
            res = layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[3, 3], stride=2 if down_sample else 1,
                                weights_initializer=tf.truncated_normal_initializer(
                                    stddev=math.sqrt(2.0 / 9.0 / num_inputs)),
                                weights_regularizer=layers.l2_regularizer(weight_decay),
                                biases_regularizer=layers.l2_regularizer(weight_decay),
                                scope='conv1', normalizer_fn=layers.batch_norm)
            res = layers.conv2d(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(
                                    stddev=math.sqrt(2.0 / 9.0 / num_outputs)),
                                weights_regularizer=layers.l2_regularizer(weight_decay),
                                biases_regularizer=layers.l2_regularizer(weight_decay),
                                scope='conv2', normalizer_fn=layers.batch_norm)
            if num_inputs != num_outputs:
                w = tf.Variable(
                    tf.truncated_normal([1, 1, num_inputs, num_outputs], stddev=math.sqrt(2.0 / num_inputs)))
                inputs = tf.nn.conv2d(inputs, w, [1, 2, 2, 1], 'SAME')
            res = tf.nn.relu(res + inputs)
        return res

    def _build_net(self, x, n):
        with tf.variable_scope('pre'):
            pre = layers.conv2d(inputs=x, num_outputs=16, kernel_size=[3, 3], scope='conv',
                                weights_initializer=tf.truncated_normal_initializer(
                                    stddev=math.sqrt(2.0 / 9.0 / 3)),
                                weights_regularizer=layers.l2_regularizer(0.0001),
                                biases_regularizer=layers.l2_regularizer(0.0001),
                                normalizer_fn=layers.batch_norm)
        h = pre
        for i in range(1, n + 1):
            h = self._block(h, 16, 0.0001, '16_block{}'.format(i))
        h = self._block(h, 32, 0.0001, '32_block1', True)
        for i in range(2, n + 1):
            h = self._block(h, 32, 0.0001, '32_block{}'.format(i))
        h = self._block(h, 64, 0.0001, '64_block1', True)
        for i in range(2, n + 1):
            h = self._block(h, 64, 0.0001, '64_block{}'.format(i))
        h = layers.conv2d(inputs=h, num_outputs=10, kernel_size=[8, 8], scope='fc1', padding='VALID',
                          weights_initializer=tf.truncated_normal_initializer(
                              stddev=math.sqrt(2.0 / 64 / 10)),
                          weights_regularizer=layers.l2_regularizer(0.0001),
                          biases_regularizer=layers.l2_regularizer(0.0001),
                          normalizer_fn=layers.batch_norm)
        self.predict = tf.reshape(h, [-1, 10])