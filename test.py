import tensorflow as tf
import tensorflow.contrib.losses as loss
import tensorflow.contrib.layers as layers
import numpy as np
import os
import pickle
import time

import modu.resnet as res

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_string('data_dir', './test/data', 'data direction')
flags.DEFINE_string('log_dir', './test/logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './test/ckpt', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_integer('decay_steps', 100, 'decay steps')
flags.DEFINE_float('decay_rate', 0.95, 'decay rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.app.flags.DEFINE_float('dropout', 0.5, 'keep probability')
tf.app.flags.DEFINE_integer('max_steps', 20000, 'max steps')

FLAGS = flags.FLAGS

def build_net(x):
    weight_decay = FLAGS.weight_decay
    h1 = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[5, 5],
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=layers.l2_regularizer(weight_decay),
                      biases_regularizer=layers.l2_regularizer(weight_decay),
                      scope='conv1', normalizer_fn=layers.batch_norm)
    h1 = layers.avg_pool2d(inputs=h1, kernel_size=[3, 3], padding='SAME', scope='pool1')

    h2 = layers.conv2d(inputs=h1, num_outputs=32, kernel_size=[5, 5],
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                      weights_regularizer=layers.l2_regularizer(weight_decay),
                      biases_regularizer=layers.l2_regularizer(weight_decay),
                      scope='conv2', normalizer_fn=layers.batch_norm)
    h2 = layers.avg_pool2d(inputs=h2, kernel_size=[3, 3], padding='SAME', scope='pool2')

    h3 = layers.conv2d(inputs=h2, num_outputs=64, kernel_size=[5, 5],
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       biases_regularizer=layers.l2_regularizer(weight_decay),
                       scope='conv3', normalizer_fn=layers.batch_norm)
    h3 = layers.avg_pool2d(inputs=h3, kernel_size=[3, 3], padding='SAME', scope='pool3')

    h4 = layers.conv2d(inputs=h3, num_outputs=64, kernel_size=[4, 4],
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       biases_regularizer=layers.l2_regularizer(weight_decay),
                       padding='VALID', scope='conv4', normalizer_fn=layers.batch_norm)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h4 = layers.dropout(inputs=h4, keep_prob=keep_prob, scope='dropout')

    h5 = layers.fully_connected(inputs=h4, num_outputs=10, activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                weights_regularizer=layers.l2_regularizer(weight_decay),
                                biases_regularizer=layers.l2_regularizer(weight_decay),
                                scope='fc1')
    h5 = tf.reshape(h5, [-1, 10])
    return h5, keep_prob


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
    data = np.reshape(data, [-1, 32, 32, 3], 'F').transpose((0, 2, 1, 3))
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

    data = np.reshape(data, [-1, 32, 32, 3], 'F').transpose((0, 2, 1, 3))
    print('load test data: test_batch')
    return data, labels





def main(_):

    if not tf.gfile.Exists(FLAGS.data_dir):
        print('data direction is not exist!')
        return -1

    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    # tf.gfile.MakeDirs(FLAGS.log_dir)

    train_data, train_labels = load_train_data()
    # name = 'cifar10_train'

    valid_data, valid_labels = load_valid_data()
    # name = 'cifar10_valid'
    train_data = (train_data - 128) / 128.0
    valid_data = (valid_data - 128) / 128.0

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'x')
        # tf.summary.image('show', x, 1)

    with tf.name_scope('label'):
        y_ = tf.placeholder(tf.int64, [None, ], 'y')

    with tf.variable_scope('net'):
        # y, keep_prob = build_net(x)
        y, keep_prob = res.build_net(x, 3)

    with tf.name_scope('scores'):
        loss.sparse_softmax_cross_entropy(y, y_, scope='cross_entropy')
        total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

        # expp = tf.exp(y)
        #
        # correct = tf.reduce_sum(tf.multiply(tf.one_hot(y_, 10), y), 1)
        #
        # total_loss = total_loss + tf.reduce_mean(tf.log(tf.reduce_sum(expp, 1)), 0) - tf.reduce_mean(correct, 0)

        tf.summary.scalar('loss', total_loss)
        # with tf.name_scope('accuracy'):
        # with tf.name_scope('correct_prediction'):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        # accuracy = tf.metrics.accuracy(labels=y_, predictions=tf.argmax(y, 1), name='accuracy')
        # tf.summary.scalar('accuracy', accuracy)

    # loss.mean_squared_error(predictions, labels, scope='l2_1')
    # loss.mean_squared_error(predictions, labels, scope='l2_2')

    # loss_collect = tf.get_collection(tf.GraphKeys.LOSSES)
    # print((tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    with tf.name_scope('train'):
        global_step = tf.Variable(1, name="global_step")
        # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
        #     global_step, FLAGS.decay_steps, FLAGS.decay_rate, True, "learning_rate")
        learning_rate = tf.train.piecewise_constant(global_step, [5000, 15000], [0.1, 0.01, 0.001])
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum).minimize(
            total_loss, global_step=global_step)
    tf.summary.scalar('lr', learning_rate)

    merged = tf.summary.merge_all()



    with tf.Session() as sess:

        saver = tf.train.Saver(name="saver")

        if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
            saver.restore(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
        train_writer.flush()
        test_writer.flush()

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

        for i in range(1, FLAGS.max_steps + 1):
            if i % 1000 == 0 and i != 1:
                time.sleep(60)

            if i % 100 == 0 and i != 0:  # Record summaries and test-set accuracy
                # start = time.clock()
                acc, summary = sess.run([accuracy, merged], feed_dict=feed_dict(False))
                # end = time.clock()
                test_writer.add_summary(summary, i)
                # print('Accuracy at step %s: %s; %f seconds' % (i, acc, end - start))

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


if __name__ == '__main__':
    tf.app.run()
