from experience.utils import *


def deepnn(x):

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 3, 32])
        variable_summaries(W_conv1, 'weights')
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1, 'biases')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = fmp(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2, 'weights')
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2, 'biases')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = fmp(h_conv2)

    with tf.name_scope("conv3"):
        W_conv3 = weight_variable([5, 5, 64, 128])
        variable_summaries(W_conv3, 'weights')
        b_conv3 = bias_variable([128])
        variable_summaries(b_conv3, 'biases')

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

        # Second pooling layer.
        h_pool3 = fmp(h_conv3)

    with tf.name_scope("conv4"):
        W_conv4 = weight_variable([5, 5, 128, 256])
        variable_summaries(W_conv4, 'weights')
        b_conv4 = bias_variable([256])
        variable_summaries(b_conv4, 'biases')

        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

        # Second pooling layer.
        h_pool4 = fmp(h_conv4)
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * 256, 1024])
        variable_summaries(W_fc1, 'weights')
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1, 'biases')

        h_pool2_flat = tf.reshape(h_pool4, [-1, 7 * 7 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([1024, 10])
        variable_summaries(W_fc2, 'weights')
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2, 'biases')

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob