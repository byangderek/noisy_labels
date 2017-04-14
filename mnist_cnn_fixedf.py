from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data import mnist

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 3000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for mini-batch SGD.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '.', 'Directory for storing data')


def train():
    # Import data
    train = mnist('train', noise_type=2, noise_ratio=0.7, gt_prior=True, is_train=True)
    val = mnist('val')
    test = mnist('test')

    with tf.Graph().as_default():
        tf.set_random_seed(23)
        # Input placeholders
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
            m = tf.placeholder(tf.float32, [10, 10], name='noise-matrix')

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])

        # We can't initialize these variables to 0 - the network will get stuck.
        def weight_variable(shape):
            """Create a weight variable with appropriate initialization."""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            """Create a bias variable with appropriate initialization."""
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        # conv1
        W_conv1 = weight_variable([5, 5, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(image_shaped_input, W_conv1) + b_conv1)

        # pool1
        h_pool1 = max_pool_2x2(h_conv1)

        # conv2
        W_conv2 = weight_variable([5, 5, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # pool2
        h_pool2 = max_pool_2x2(h_conv2)

        # fc1
        W_fc1 = weight_variable([7 * 7 * 32, 128])
        b_fc1 = bias_variable([128])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # fc1_dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # fc2_output_softmax
        W_fc2 = weight_variable([128, 10])
        b_fc2 = bias_variable([10])
        y1 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        y2 = tf.matmul(y1, m)

        # cross-entropy loss
        with tf.name_scope('cross_entropy'):
            diff = y_ * tf.log(y2)
            with tf.name_scope('total'):
                cross_entropy = -tf.reduce_mean(diff)

        # optimizer
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
                cross_entropy)

        # compute classification accuracy
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction1 = tf.equal(tf.argmax(y1, 1), tf.argmax(y_, 1))
                correct_prediction2 = tf.equal(tf.argmax(y2, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
                accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(init)

            def feed_dict(dataset, batch_size, train=False):
                xs, ys = dataset.get_data(batch_size)
                if train:
                    k = FLAGS.dropout
                else:
                    k = 1.0
                return {x: xs, y_: ys, keep_prob: k, m: dataset._noise_matrix}

            for i in range(FLAGS.max_steps):
                if i % 200 == 0:  # Record summaries and test-set accuracy
                    acc = sess.run(accuracy1, feed_dict=feed_dict(val, 0))
                    print('Validation accuracy at step %s: %s' % (i, acc))
                else:  # Record train set summaries, and train
                    loss, acc, _ = sess.run([cross_entropy, accuracy2, train_step], feed_dict=feed_dict(train, FLAGS.batch_size, True))
                    if i % 20 == 0:
                        print('Iter %d: loss %f, acc %.2f%%' % (i, loss, 100 * acc))

            print('Final test accuracy %s' % (sess.run(accuracy1, feed_dict=feed_dict(test, 0))))


if __name__ == '__main__':
    train()
