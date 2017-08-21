import tensorflow as tf
import tensorflow.contrib.losses as loss
import tensorflow.contrib.layers as layers
import os
import shutil

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.05, 'learning rate')
flags.DEFINE_string('data_dir', './test/data', 'data direction')
flags.DEFINE_string('log_dir', './test/logs', 'log direction')
flags.DEFINE_float('weight_decay', 0.01, 'weight decay')
flags.DEFINE_integer('decay_steps', 7500, 'decay steps')
flags.DEFINE_float('decay_rate', 1, 'decay rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
FLAGS = flags.FLAGS

def build_net(x):
    weight_decay = FLAGS.weight_decay
    h1 = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[5, 5],
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=layers.l2_regularizer(weight_decay),
                      biases_regularizer=layers.l2_regularizer(weight_decay),
                      scope='conv1')
    h1 = layers.avg_pool2d(inputs=h1, kernel_size=[3, 3], padding='SAME', scope='pool1')

    h2 = layers.conv2d(inputs=h1, num_outputs=32, kernel_size=[5, 5],
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                      weights_regularizer=layers.l2_regularizer(weight_decay),
                      biases_regularizer=layers.l2_regularizer(weight_decay),
                      scope='conv2')
    h2 = layers.avg_pool2d(inputs=h2, kernel_size=[3, 3], padding='SAME', scope='pool2')

    h3 = layers.conv2d(inputs=h2, num_outputs=64, kernel_size=[5, 5],
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       biases_regularizer=layers.l2_regularizer(weight_decay),
                       scope='conv3')
    h3 = layers.avg_pool2d(inputs=h3, kernel_size=[3, 3], padding='SAME', scope='pool3')

    h4 = layers.conv2d(inputs=h3, num_outputs=64, kernel_size=[4, 4],
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       biases_regularizer=layers.l2_regularizer(weight_decay),
                       padding='VALID', scope='conv4')
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h4 = layers.dropout(inputs=h4, keep_prob=keep_prob, scope='dropout')

    h5 = layers.fully_connected(inputs=h4, num_outputs=10, activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
                                weights_regularizer=layers.l2_regularizer(weight_decay),
                                biases_regularizer=layers.l2_regularizer(weight_decay),
                                scope='fc1')
    h5 = tf.reshape(h5, [-1, 10])
    return h5, keep_prob

def main(_):

    if not os.path.exists(FLAGS.data_dir):
        print('data direction is not exist!')
        return -1

    if os.path.exists(FLAGS.log_dir):
        shutil.rmtree(FLAGS.log_dir)
    os.mkdir(FLAGS.log_dir)
    predictions = tf.constant([1, 2, 4], tf.float32, [1, 3], 'predicted_label')
    labels = tf.constant([2, 1, 3], tf.float32, [1, 3], 'ground_truth')

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'x')
        y_ = tf.placeholder(tf.int64, [None, ], 'y')

    # with tf.name_scope('net'):
    y, keep_prob = build_net(x)
    loss.sparse_softmax_cross_entropy(y, y_, scope='cross_entropy')
    # loss.mean_squared_error(predictions, labels, scope='l2_1')
    # loss.mean_squared_error(predictions, labels, scope='l2_2')

    # loss_collect = tf.get_collection(tf.GraphKeys.LOSSES)
    total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True, name='total_loss')
    # print((tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    with tf.name_scope('train'):
        global_step = tf.Variable(0, name="global_step")
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
            global_step, FLAGS.decay_steps, FLAGS.decay_rate, True, "learning_rate")
        train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum).minimize(
            total_loss, global_step=global_step)
    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=y_, predictions=tf.argmax(y, 1), name='accuracy')

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        writer.flush()

        writer.close()


if __name__ == '__main__':
    tf.app.run()