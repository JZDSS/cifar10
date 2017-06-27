import tensorflow as tf


tf.app.flags.DEFINE_integer('-epochs', 10, 'number of epochs')
tf.app.flags.DEFINE_float('-learning rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_string('-data', '../data', 'data set direction')
FLAGS = tf.app.flags.FLAGS


def main(_):
    print(0)
    return 0

if __name__ == '__main__':
    tf.app.run(main)
