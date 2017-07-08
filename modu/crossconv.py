from modu.utils import *

def deepnn(x):
    with tf.name_scope('block1'):
        conv1_1 = tf.contrib.layers.conv2d(x, 10, [3, 1], scope='conv1_1')
        pool1_1 = tf.contrib.layers.max_pool2d(conv1_1, [2, 2], scope='pool1_1')

        conv1_2 = tf.contrib.layers.conv2d(pool1_1, 10, [1, 3], scope='conv1_2')
        pool1_2 = tf.contrib.layers.max_pool2d(conv1_2, [2, 2], scope='pool1_2')

        conv2_1 = tf.contrib.layers.conv2d(x, 10, [1, 3], scope='conv2_1')
        pool2_1 = tf.contrib.layers.max_pool2d(conv2_1, [2, 2], scope='pool2_1')

        conv2_2 = tf.contrib.layers.conv2d(pool2_1, 10, [3, 1], scope='conv2_2')
        pool2_2 = tf.contrib.layers.max_pool2d(conv2_2, [2, 2], scope='pool2_2')

        s = tf.stack([pool1_2, pool2_2], axis=1)

    conv3 = tf.contrib.layers.conv2d(s, 20, [1, 1])
    pool3 = tf.contrib.layers.max_pool2d(conv3, [2, 2])

    conv4 = tf.contrib.layers.conv2d(pool3, 40, [1, 1])
    pool4 = tf.contrib.layers.max_pool2d(conv4, [2, 2])

    conv5 = tf.contrib.layers.conv2d(pool4, 80, [1, 1])
    pool5 = tf.contrib.layers.max_pool2d(conv5, [2, 2])
    fc1 = tf.contrib.layers.fully_connected(pool5, 160)
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        fc1_drop = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.contrib.layers.fully_connected(fc1_drop, 10, activation_fn=None)
    fc2 = tf.reshape(fc2, [-1, 10])
    return fc2, keep_prob


def deepnn2(x):

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.variable_scope("conv1"):
        with tf.variable_scope('1'):
            W_conv1_1 = weight_variable([5, 1, 3, 32])
            variable_summaries(W_conv1_1, 'weights')
            b_conv1_1 = bias_variable([32])
            variable_summaries(b_conv1_1, 'biases')
            h1_1 = tf.nn.relu(conv2d(x, W_conv1_1) + b_conv1_1)
            p1_1 = max_pool_2x2(h1_1)

        with tf.variable_scope('2'):
            W_conv1_2 = weight_variable([1, 5, 3, 32])
            variable_summaries(W_conv1_2, 'weights')
            b_conv1_2 = bias_variable([32])
            variable_summaries(b_conv1_2, 'biases')
            h1_2 = tf.nn.relu(conv2d(x, W_conv1_2) + b_conv1_2)
            p1_2 = max_pool_2x2(h1_2)


    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.variable_scope("conv2"):
        with tf.variable_scope('1'):
            W_conv2_1 = weight_variable([1, 5, 32, 64])
            variable_summaries(W_conv2_1, 'weights')
            b_conv2_1 = bias_variable([64])
            variable_summaries(b_conv2_1, 'biases')
            h2_1 = tf.nn.relu(conv2d(p1_1, W_conv2_1) + b_conv2_1)
            p2_1 = max_pool_2x2(h2_1)

        with tf.variable_scope('2'):
            W_conv2_2 = weight_variable([5, 1, 32, 64])
            variable_summaries(W_conv2_2, 'weights')
            b_conv2_2 = bias_variable([64])
            variable_summaries(b_conv2_2, 'biases')
            h2_2 = tf.nn.relu(conv2d(p1_2, W_conv2_2) + b_conv2_2)
            p2_2 = max_pool_2x2(h2_2)

    h_pool2 = p2_1 + p2_2

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.variable_scope("fc1"):
        W_fc1 = weight_variable([8 * 8 * 64, 1024])
        variable_summaries(W_fc1, 'weights')
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1, 'biases')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.variable_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.variable_scope("fc2"):
        W_fc2 = weight_variable([1024, 10])
        variable_summaries(W_fc2, 'weights')
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2, 'biases')

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob