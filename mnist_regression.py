# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
#matplotlib import to be able to plot graphs
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import mnist data set built in tensorflow
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS= None
VALIDATION_SIZE = 2000
#==============================================================================
#The main function get the dataset and converts its labels to an array of size 10
#using onehot. It implements the linear model. It calculates the cross entropy.
#Backpropagation is used to minimize the cross entropy. It trains for 2000 steps
# and produces a graph of accuracy rates vs number of steps.
#==============================================================================
def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

    ##X is just a placeholder, a value that we have to enter when we run computations
    #The shape is [None, 784]
    x = tf.placeholder(tf.float32, [None, 784])
    ##Weight - this is a modifiable tensor of shape [784, 10]. Initialised to zeros
    W = tf.Variable(tf.zeros([784, 10])) 
    ##Bias - this is a modifiable tensor of shape [10]. Initialised to zeros
    b = tf.Variable(tf.zeros([10])) #bias
    
    
    ##implementing our linear model: Y = XW + b
    y = tf.nn.softmax(tf.matmul(x, W)+b)

    ##A placeholder for the correct answers
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    ##implement cross entropy. Cross Entropy is the measure of how inefficient
    ##our predictions are for describing the truth
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

    ##Doing backprop to minimize the cross entropy
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    ##Launch a model in an interactive session
    sess = tf.InteractiveSession()

    ##initialise the variables
    tf.global_variables_initializer().run()
    
    ##visualization variables
    train_accuracies =[]
    validation_accuracies= []
    x_range = []

    display_step =1
    ##train for 20000 steps
    #For each step 100 random data points are chosen from training set
    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        if i%display_step ==0 or (i+1) == 2000:
            ##prediction: check if our prediction matches with the truth
            correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(y_, 1))
            #checking the accuracy of the prediction
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            train_accuracy = accuracy.eval(feed_dict = {x:batch_xs,
                                                        y_: batch_ys})
            if(VALIDATION_SIZE):
                validation_accuracy = accuracy.eval(feed_dict = {x: mnist.test.images[0:50],
                                                                 y_: mnist.test.labels[0:50]})

                print('training_accuracy /validation_accuracy => %.2f /%.2f for step %d' %(train_accuracy, validation_accuracy, i))
                validation_accuracies.append(validation_accuracy) ##validation accuracies for plotting

            else:
                print('training_accuracy => %.4f for step %d' %(train_accuracy, i))
            train_accuracies.append(train_accuracy)
            x_range.append(i) ## add i to the x range array for plotting later

            ##increase display_step
            if i%(display_step *10) ==0 and i:
                display_step *=10

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    validation_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
#========================Plotting============================================
    print('validation_accuracy => %.4f' %validation_accuracy)
    plt.plot(x_range, train_accuracies, '-b', label ='Training')
    plt.plot(x_range, validation_accuracies, '-g', label='Validation')
    plt.legend(loc ='lower_right', frameon=False)
    plt.ylim(ymax= 1.1, ymin =0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.savefig('regression.jpg')
#============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default ='../MNIST-data', help = 'Directory fro storing input datat')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]]+ unparsed)
