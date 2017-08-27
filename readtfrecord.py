import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


for serialized_example in tf.python_io.tf_record_iterator("./data/cifar10_train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image_raw'].bytes_list.value[0]
    label = example.features.feature['label'].int64_list.value
    im = np.fromstring(image, dtype=np.uint8)
    im = np.reshape(im, [32, 32, 3])
    plt.imshow(im)
    plt.show()
    # cv2.waitKey(0)
    print(image, label)

# reader = tf.python_io.tf_record_iterator('./data/cifar10_train.tfrecords')
# those_examples = [tf.train.Example().FromString(example_str)
#                   for example_str in reader]
#
# a = 1


# reader = tf.python_io.tf_record_iterator('./data/cifar10_train.tfrecords')
# those_examples = [tf.train.Example().FromString(example_str)
#                   for example_str in reader]
# same_example = those_examples[0]
#
# same_image_bytes = same_example.features.feature['image_raw'].bytes_list.value[0]
#
# im = np.fromstring(same_image_bytes, dtype=np.uint8)
# im = np.reshape(im, [32, 32, 3], 'C')
#
# plt.imshow(im)
# plt.show()
# a = 1