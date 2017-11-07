import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def read_from_tfrecord(tfrecord_file_queue):
    # tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }, name='features')
    image = tf.decode_raw(tfrecord_features['image_raw'], tf.uint8)
    ground_truth = tf.decode_raw(tfrecord_features['label'], tf.int32)

    image = tf.reshape(image, [32, 32, 3])
    ground_truth = tf.reshape(ground_truth, [1])
    return image, ground_truth

image, label = read_from_tfrecord(tf.train.string_input_producer(['data/cifar10_train.tfrecords']))
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    image, labels = sess.run([image, label])
    print(labels)
    plt.imshow(image)
    plt.show()
    coord.request_stop()
    coord.join(threads)
