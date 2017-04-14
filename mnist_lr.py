from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from data import mnist

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 3000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for mini-batch SGD.')
flags.DEFINE_string('data_dir', '.', 'Directory for storing data')
flags.DEFINE_float('noise_ratio', 0.3, 'Label noise ratio.')

def train():
    # Import data
    train = mnist('train', noise_type=2, noise_ratio=FLAGS.noise_ratio, is_train=True)
    train0 = mnist('train')
    val = mnist('val')
    test = mnist('test')

    with tf.Graph().as_default():
        tf.set_random_seed(23)
        # Input placeholders
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

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

        def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
            """Reusable code for making a simple neural net layer.

            It does a matrix multiply, bias add, and then uses relu to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim])
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, weights) + biases
                activations = act(preactivate, name='activation')
                return activations

        # fc1
        y = nn_layer(x, 784, 10, 'layer1', act=tf.nn.softmax)

        # cross-entropy loss
        with tf.name_scope('cross_entropy'):
            diff = y_ * tf.log(y)
            with tf.name_scope('total'):
                cross_entropy = -tf.reduce_mean(diff)

        # optimizer
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
                cross_entropy)

        # compute classification accuracy
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(init)

            def feed_dict(dataset, batch_size):
                """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
                xs, ys = dataset.get_data(batch_size)
                return {x: xs, y_: ys}

            for i in range(FLAGS.max_steps):
                if i % 200 == 0:  # Record summaries and test-set accuracy
                    acc = sess.run(accuracy, feed_dict=feed_dict(val, 0))
                    print('Validation accuracy at step %d: %.2f%%' % (i, acc*100))
                else:  # Record train set summaries, and train
                    loss, acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict=feed_dict(train, FLAGS.batch_size))
                    if i % 20 == 0:
                        print('Iter %d: loss %f, acc %.2f%%' % (i, loss, 100*acc))

            print('Final test accuracy %s' % (sess.run(accuracy, feed_dict=feed_dict(test, 0))))
            # compute the confusion matrix on train set to for later initialization use
            ll = sess.run(y, feed_dict=feed_dict(train0, 0))
            ll = np.argmax(ll, axis=1)
            m = confusion_matrix(train0.labels, ll)
            ms = np.sum(m, axis=1)
            m = m/ms[:,np.newaxis]
            np.save('weights/lr_m_%.1f.npy' % FLAGS.noise_ratio, m)

if __name__ == '__main__':
    train()
