import os
import pickle
import time

import numpy as np
import tensorflow as tf

from modu import allconv
from modu import baseline
from modu import fmp
from modu import crossconv
from modu import matconvnet

tf.app.flags.DEFINE_integer('-epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_float('-learning_rate', 0.002, 'learning rate')
tf.app.flags.DEFINE_string('-data_dir', './data', 'data set direction')
tf.app.flags.DEFINE_string('-log_dir', './logs', 'logs direction')
tf.app.flags.DEFINE_string('-ckpt_dir', './ckpt', 'check point direction')
tf.app.flags.DEFINE_integer('-decay_steps', 100, 'decay steps')
tf.app.flags.DEFINE_float('-decay_rate', 0.97, 'decay rate')
tf.app.flags.DEFINE_integer('-batch_size', 128, 'batch size')
tf.app.flags.DEFINE_float('-dropout', 0.5, 'keep probability')
tf.app.flags.DEFINE_integer('-max_steps', 10000, 'max steps')
tf.app.flags.DEFINE_string('-model', 'matconvnet', 'baseline, fmp, allconv, cross, matconvnet')
tf.app.flags.DEFINE_bool('-lsuv', False, 'if use lsuv initialization')
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

    # tf.summary.image('show', x, 10)

    if FLAGS.model is 'baseline':
        print('baseline')
        y, keep_prob, l, name, shape = baseline.deepnn(x)
    elif FLAGS.model is 'fmp':
        print('fmp')
        y, keep_prob = fmp.fmp(x)
    elif FLAGS.model is 'allconv':
        print('allconv')
        y, keep_prob = allconv.allconv(x)
    elif FLAGS.model is 'cross':
        print('cross')
        y, keep_prob = crossconv.deepnn2(x)
    elif FLAGS.model is 'matconvnet':
        print('matconvnet')
        y, keep_prob = matconvnet.deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_, logits=y), name="cross_entropy")
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        global_step = tf.Variable(0, name="global_step")
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
            global_step, FLAGS.decay_steps, 0.95, True, "learning_rate")
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(
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

    if FLAGS.lsuv and FLAGS.model is 'baseline':
        def init(layers, image, names, shapes):
            n_max = 5
            i = -1
            for layer in layers:
                i = i + 1
                mean = tf.reduce_mean(layer)
                std = tf.sqrt(tf.reduce_mean(tf.square(layer - mean)))
                n = 0
                std_div = sess.run(std, feed_dict={x: image, keep_prob: FLAGS.dropout})
                while abs(std_div - 1.0) > 0.01 and n < n_max:
                    n = n + 1
                    with tf.variable_scope(names[i], reuse=True):
                        w = tf.get_variable('weights', shape=shapes[i])
                        update = tf.assign(w, w / std_div)
                        sess.run(update)
                    std_div = sess.run(std, feed_dict={x: image, keep_prob: FLAGS.dropout})

        init(l, train_data[np.random.randint(low=0, high=train_labels.shape[0], size=FLAGS.batch_size, dtype=np.int32), ...],
                  name, shape)

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
            time.sleep(300)

        if i % 100 == 0 and i != 0:  # Record summaries and test-set accuracy
            start = time.clock()
            acc, summary = sess.run([accuracy, merged], feed_dict=feed_dict(False))
            end = time.clock()
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s; %f seconds' % (i, acc, end - start))

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
