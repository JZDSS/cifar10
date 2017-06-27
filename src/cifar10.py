import tensorflow as tf
import pickle
import os
import numpy as np


tf.app.flags.DEFINE_integer('-epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_float('-learning rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_string('-data', '../data', 'data set direction')
FLAGS = tf.app.flags.FLAGS


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_raining_data():
    data = np.ndarray(shape=(0, 32*32*3), dtype=float)
    for i in range(5):
        tmp = unpickle(os.path.join(FLAGS.data, "data_batch_{}".format(i + 1)))
        print(tmp)
        data = np.append(data, tmp["data"], axis=0)
        print(data.shape)


def main(_):
    load_raining_data()
    print(0)
    return 0

if __name__ == '__main__':
    tf.app.run(main)
