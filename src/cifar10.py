import tensorflow as tf
import pickle
import os
import numpy as np


tf.app.flags.DEFINE_integer('-epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_float('-learning rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_string('-data', '../data', 'data set direction')
tf.app.flags.DEFINE_string('-logs', '../logs', 'logs direction')
FLAGS = tf.app.flags.FLAGS


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data():
    data = np.ndarray(shape=(0, 32*32*3), dtype=np.float32)
    labels = np.ndarray(shape=0, dtype=np.int64)
    for i in range(5):
        tmp = unpickle(os.path.join(FLAGS.data, "data_batch_{}".format(i + 1)))
        data = np.append(data, tmp[b'data'], axis=0)
        labels = np.append(labels, tmp[b'labels'], axis=0)
        print('load training data: data_batch_{}'.format(i + 1))
    return data, labels


def load_valid_data():
    tmp = unpickle(os.path.join(FLAGS.data, "test_batch"))
    data = tmp[b'data']
    labels = np.ndarray(shape=(0, 1), dtype=np.int64)
    labels = np.append(labels, tmp[b'labels'])
    data.astype(np.float32)
    data.astype(np.int64)
    return data, labels


def main(_):
    train_data, train_labels = load_train_data()
    valid_data, valid_labels = load_valid_data()
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [-1, 32*32*3], 'x')
        y_ = tf.placeholder(tf.int64, [-1, ], 'y')


    return 0

if __name__ == '__main__':
    tf.app.run(main)
