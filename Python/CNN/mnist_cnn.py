import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

# Define a CNN with 2 convolution layer
# conv2D -> conv2D -> fc -> dropout -> fc -> softmax

def main():

    mnist = input_data.read_data_sets('../dataset', one_hot=True)

    # convolution weight and bias define
    conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32],stddev=0.1,seed=SEED, dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
    
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1,seed=SEED, dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))

    # define a 512 depth FC layer
    fc1_weights = tf.Variable(tf.truncated_normal([7 * 7 * 64, 512], stddev=0.1, seed=SEED, dtype=tf.float32))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))

    # another FC layer, after dropout
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32))

    # model closure
    def model(data_set):
        x_image = tf.reshape(data_set, shape=[-1, 28, 28, 1])
        # now construct all layer

        # START: convolution layer 1
        conv = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME', name='Conv2D1')

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # START: convolution layer 2
        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME', name='Conv2D2')

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # before feed tensor to FC layer, we need to reshape pooling output
        pool_shape = pool.get_shape().as_list()
        # arg shape could not start with None Type, thus use -1 rather than data[0]
        reshape = tf.reshape(pool, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        # add hidden layer, 50% dropout rate
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # define input X and output label
    y_label = tf.placeholder(tf.int64, shape=[None, 10])
    x_train = tf.placeholder(tf.float32, shape=[None, 784])

    # get model output
    logits = model(x_train)

    # you can use below evaluation to replace tf.nn.softmax_cross_entropy_with_logits
    # -tf.reduce_sum(y_label * tf.log(logits), reduction_indices=[1])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=logits))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # result as a list with value like [1, 0, 0, 1 ...]
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_label, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x_train: batch[0], y_label: batch[1]})
                print('step %d, accuracy %g'%(i, train_accuracy))
            train_step.run(feed_dict={x_train: batch[0], y_label: batch[1]})

        print("test accuracy %g"%accuracy.eval(feed_dict={x_train: mnist.test.images, y_label: mnist.test.labels}))




if __name__ == '__main__':
    main()
