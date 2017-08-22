import os
import pickle
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# import cv2

flags = tf.app.flags
flags.DEFINE_string('data_dir', './test/data', 'data direction')
flags.DEFINE_string('outdirectory', './test/', '')
flags.DEFINE_bool('train', True, 'True for training data, False for valid data')
FLAGS = flags.FLAGS

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


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(_):

    if FLAGS.train:
        data, labels = load_train_data()
        name = 'cifar10_train'
    else:
        data, labels = load_valid_data()
        name = 'cifar10_valid'
    # data = (data - 128) / 128.0
    # valid_data = (valid_data - 128) / 128.0



    num_examples = data.shape[0]
    rows = data.shape[1]
    cols = data.shape[2]
    depth = data.shape[3]
    filename = os.path.join(FLAGS.outdirectory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = data[index].tostring()
        d = data[index].astype(np.uint8)
        # plt.imshow(d)
        # plt.show()
        # cv2.imshow('a', d)
        # cv2.waitKey(0)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.app.run()