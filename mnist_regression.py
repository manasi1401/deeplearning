from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
FLAGS= None

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10])) #weight
    b = tf.Variable(tf.zeros([10])) #bias

#implement model
    y = tf.matmul(x, W)+b

#train
    y_ = tf.placeholder(tf.float32, [None, 10])
    #implement cros entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    #initialise the variables
    tf.global_variables_initializer().run()

    #train for 1000 steps
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #prediction
    correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy: ")
    print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default ='../MNIST-data', help = 'Directory fro storing input datat')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]]+ unparsed)
