import tensorflow as tf
import pickle
import os
import numpy as np
import time


tf.app.flags.DEFINE_integer('-epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_float('-learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_string('-data_dir', '../data', 'data set direction')
tf.app.flags.DEFINE_string('-log_dir', '../logs', 'logs direction')
tf.app.flags.DEFINE_string('-ckpt_dir', '../ckpt', 'check point direction')
tf.app.flags.DEFINE_integer('-decay_steps', 100, 'decay steps')
tf.app.flags.DEFINE_integer('-batch_size', 128, 'batch size')
tf.app.flags.DEFINE_float('-dropout', 0.5, 'keep probability')
tf.app.flags.DEFINE_integer('-max_steps', 10000, 'max steps')
FLAGS = tf.app.flags.FLAGS


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data():
    data = np.ndarray(shape=(0, 32 * 32 * 3), dtype=np.float32)
    labels = np.ndarray(shape=0, dtype=np.int64)
    for i in range(5):
        tmp = unpickle(os.path.join(FLAGS.data_dir, "data_batch_{}".format(i + 1)))
        data = np.append(data, tmp[b'data'], axis=0)
        labels = np.append(labels, tmp[b'labels'], axis=0)
        print('load training data: data_batch_{}'.format(i + 1))
    data = np.reshape(data, [-1, 32, 32, 3], 'F')
    return data, labels


def load_valid_data():
    tmp = unpickle(os.path.join(FLAGS.data_dir, "test_batch"))
    data = np.ndarray(shape=(0, 32 * 32 * 3), dtype=np.float32)
    labels = np.ndarray(shape=0, dtype=np.int64)

    data = np.append(data, tmp[b'data'], axis=0)
    # data = tmp[b'data']
    labels = np.append(labels, tmp[b'labels'])

    # data.astype(np.float32)
    # labels.astype(np.int64)

    data = np.reshape(data, [-1, 32, 32, 3], 'F')
    print('load test data: test_batch')
    return data, labels


def deepnn(x):
    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="weights")

    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="biases")

    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            with tf.name_scope(name):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 3, 32])
        variable_summaries(W_conv1, 'weights')
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1, 'biases')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2, 'weights')
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2, 'biases')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([8 * 8 * 64, 1024])
        variable_summaries(W_fc1, 'weights')
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1, 'biases')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
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


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train_data, train_labels = load_train_data()
    valid_data, valid_labels = load_valid_data()
    train_data = (train_data - 128) / 128.0
    valid_data = (valid_data - 128) / 128.0

    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'x')
        y_ = tf.placeholder(tf.int64, [None, ], 'y')

    tf.summary.image('show', x, 10)

    y, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_, logits=y), name="cross_entropy")
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        global_step = tf.Variable(0, name="global_step")
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
            global_step, FLAGS.decay_steps, 0.95, True, "learning_rate")
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            cross_entropy, global_step=global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.name_scope("saver"):
        saver = tf.train.Saver(name="saver")

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
    train_writer.flush()
    test_writer.flush()

    if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
        saver.restore(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
        # acc = sess.run(accuracy, feed_dict={x: train_data[1:5000, ...], y_: train_labels[1:5000], keep_prob: 1.0})
        # print(acc)
    else:
        tf.global_variables_initializer().run()

    def feed_dict(train, kk=FLAGS.dropout):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        def get_batch(data, labels):
            id = np.random.randint(low=0, high=labels.shape[0], size=FLAGS.batch_size, dtype=np.int32)
            return data[id, ...], labels[id]
        if train:
            xs, ys = get_batch(train_data, train_labels)
            k = kk
        else:
            # xs, ys = get_batch(valid_data, valid_labels)
            xs = valid_data
            ys = valid_labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps + 1):
        if i % 1000 == 0 and i != 0:
            time.sleep(100)

        if i % 100 == 0:  # Record summaries and test-set accuracy
            acc, summary = sess.run([accuracy, merged], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))

        # else:  # Record train set summaries, and train
        if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            feed = feed_dict(True)
            sess.run(train_step,
                  feed_dict=feed,
                  options=run_options,
                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)

            feed[keep_prob] = 1.0
            summary = sess.run(merged, feed_dict=feed)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for step', i)
            saver.save(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
        else:  # Record a summary
            feed = feed_dict(True)
            sess.run(train_step, feed_dict=feed)

            feed[keep_prob] = 1.0
            summary = sess.run(merged, feed_dict=feed)
            train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
    return 0

if __name__ == '__main__':
    tf.app.run(main)
