from experience.utils import *

weight_decay = tf.constant(0.0001)

def deepnn(x):
    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable_v2([5, 5, 3, 32], 0.01, weight_decay)
        # variable_summaries(W_conv1, 'weights')
        b_conv1 = bias_variable_v2([32], 0.0, weight_decay)
        # variable_summaries(b_conv1, 'biases')
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool32(h_conv1)

    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable_v2([5, 5, 32, 32], 0.05, weight_decay)
        # variable_summaries(W_conv2, 'weights')
        b_conv2 = bias_variable_v2([32], 0.0, weight_decay)
        # variable_summaries(b_conv2, 'biases')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = avg_pool32(h_conv2)

    with tf.variable_scope('conv3'):
        W_conv3 = weight_variable_v2([5, 5, 32, 64], 0.05, weight_decay)
        # variable_summaries(W_conv3, 'weights')
        b_conv3 = bias_variable_v2([64], 0.0, weight_decay)
        # variable_summaries(b_conv3, 'biases')
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = avg_pool32(h_conv3)

    with tf.variable_scope('conv4'):
        W_conv4 = weight_variable_v2([4, 4, 64, 64], 0.05, weight_decay)
        # variable_summaries(W_conv4, 'weights')
        b_conv4 = bias_variable_v2([64], 0.0, weight_decay)
        # variable_summaries(b_conv4, 'biases')
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, [1, 1, 1, 1], 'VALID') + b_conv4)

    with tf.variable_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_conv4 = tf.nn.dropout(h_conv4, keep_prob)

    with tf.variable_scope('conv5'):
        W_conv5 = weight_variable_v2([1, 1, 64, 10], 0.05, weight_decay)
        # variable_summaries(W_conv5, 'weights')
        b_conv5 = bias_variable_v2([10], 0.0, weight_decay)
        # variable_summaries(b_conv5, 'biases')
        h_conv5 = conv2d(h_conv4, W_conv5) + b_conv5

    h_conv5 = tf.reshape(h_conv5, [-1, 10])
    return h_conv5, keep_prob


def deepnn2(x):
    with tf.variable_scope("conv1"):
        with tf.variable_scope('1'):
            W_conv1_1 = weight_variable([5, 1, 3, 32])
            # variable_summaries(W_conv1_1, 'weights')
            b_conv1_1 = bias_variable([32])
            # variable_summaries(b_conv1_1, 'biases')
            h1_1 = tf.nn.relu(conv2d(x, W_conv1_1) + b_conv1_1)
            p1_1 = max_pool_2x2(h1_1)

        with tf.variable_scope('2'):
            W_conv1_2 = weight_variable([1, 5, 3, 32])
            # variable_summaries(W_conv1_2, 'weights')
            b_conv1_2 = bias_variable([32])
            # variable_summaries(b_conv1_2, 'biases')
            h1_2 = tf.nn.relu(conv2d(x, W_conv1_2) + b_conv1_2)
            p1_2 = max_pool_2x2(h1_2)


    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.variable_scope("conv2"):
        with tf.variable_scope('1'):
            W_conv2_1 = weight_variable([1, 5, 32, 32])
            # variable_summaries(W_conv2_1, 'weights')
            b_conv2_1 = bias_variable([32])
            # variable_summaries(b_conv2_1, 'biases')
            h2_1 = tf.nn.relu(conv2d(p1_1, W_conv2_1) + b_conv2_1)
            p2_1 = max_pool_2x2(h2_1)

        with tf.variable_scope('2'):
            W_conv2_2 = weight_variable([5, 1, 32, 32])
            # variable_summaries(W_conv2_2, 'weights')
            b_conv2_2 = bias_variable([32])
            # variable_summaries(b_conv2_2, 'biases')
            h2_2 = tf.nn.relu(conv2d(p1_2, W_conv2_2) + b_conv2_2)
            p2_2 = max_pool_2x2(h2_2)

    h_pool2 = p2_1 + p2_2

    with tf.variable_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 32, 64])
        # variable_summaries(W_conv3, 'weights')
        b_conv3 = bias_variable([64])
        # variable_summaries(b_conv3, 'biases')
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool32(h_conv3)

    with tf.variable_scope('conv4'):
        W_conv4 = weight_variable([4, 4, 64, 64])
        # variable_summaries(W_conv4, 'weights')
        b_conv4 = bias_variable([64])
        # variable_summaries(b_conv4, 'biases')
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, [1, 1, 1, 1], 'VALID') + b_conv4)

    with tf.variable_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_conv4 = tf.nn.dropout(h_conv4, keep_prob)

    with tf.variable_scope('conv5'):
        W_conv5 = weight_variable([1, 1, 64, 10])
        # variable_summaries(W_conv5, 'weights')
        b_conv5 = bias_variable([10])
        # variable_summaries(b_conv5, 'biases')
        h_conv5 = conv2d(h_conv4, W_conv5) + b_conv5
    h_conv5 = tf.reshape(h_conv5, [-1, 10])
    return h_conv5, keep_prob