from src.utils import *

def deepnn(x):

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 3, 32])
        variable_summaries(W_conv1, 'weights')
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1, 'biases')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool32(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2, 'weights')
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2, 'biases')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = max_pool32(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([1, 1, 64, 1024])
        variable_summaries(W_fc1, 'weights')
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1, 'biases')

        h_fc1 = conv2d(h_pool2, W_fc1) + b_fc1

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([1, 1, 1024, 10])
        variable_summaries(W_fc2, 'weights')
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2, 'biases')

        y_conv = conv2d(h_fc1_drop, W_fc2) + b_fc2

        y_conv = tf.reduce_mean(y_conv, [1, 2])
    return y_conv, keep_prob


def convpool(x):

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 3, 32])
        variable_summaries(W_conv1, 'weights')
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1, 'biases')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool32(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2, 'weights')
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2, 'biases')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = max_pool32(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([1, 1, 64, 1024])
        variable_summaries(W_fc1, 'weights')
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1, 'biases')

        h_fc1 = conv2d(h_pool2, W_fc1) + b_fc1

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([1, 1, 1024, 10])
        variable_summaries(W_fc2, 'weights')
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2, 'biases')

        y_conv = conv2d(h_fc1_drop, W_fc2) + b_fc2

        y_conv = tf.reduce_mean(y_conv, [1, 2])
    return y_conv, keep_prob


def allconv(x):

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 3, 32])
        variable_summaries(W_conv1, 'weights')
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1, 'biases')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    with tf.name_scope("pool1"):
        # Pooling layer - downsamples by 2X.
        W_pool1 = weight_variable([3, 3, 32, 32])
        variable_summaries(W_pool1, 'weights')
        b_pool1 = bias_variable([32])
        variable_summaries(b_pool1, 'biases')
        h_pool1 = conv(h_conv1, W_pool1, [1, 2, 2, 1]) + b_pool1

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2, 'weights')
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2, 'biases')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope("pool2"):
        # Pooling layer - downsamples by 2X.
        W_pool2 = weight_variable([3, 3, 64, 64])
        variable_summaries(W_pool2, 'weights')
        b_pool2 = bias_variable([64])
        variable_summaries(b_pool2, 'biases')
        h_pool2 = conv(h_conv2, W_pool2, [1, 2, 2, 1]) + b_pool2

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([1, 1, 64, 1024])
        variable_summaries(W_fc1, 'weights')
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1, 'biases')

        h_fc1 = conv2d(h_pool2, W_fc1) + b_fc1

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([1, 1, 1024, 10])
        variable_summaries(W_fc2, 'weights')
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2, 'biases')

        y_conv = conv2d(h_fc1_drop, W_fc2) + b_fc2

        y_conv = tf.reduce_mean(y_conv, [1, 2])
    return y_conv, keep_prob