# This is MNIST cnn model as described on Tensor flow tutorials with estimators
#These imports are necessary to make python 2 interpreter behave as cloes to Python 2 as possible
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#imports
#NumPy - fundamental package for array, functions, algebra, transforms
import numpy as np
#Open source machine learning framework used for this Nerual Network
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#Application logic goes below

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    #input layer. For CCN 2-D image is expected to have a
    #shape [batch_size, image_width, _image_height, channels]
    # -1 batch size indicates that it is dynamically computed based on input values in 
    #features["x"]. This makes batch_size a hyperparameter
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    #convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters=32,             #Number of filters to apply
        kernel_size = [5,5],    #filter dimensions [width, height]
        padding ="same",        #Tensor flow adds 0 at the edge to perserve width & height 27
        activation = tf.nn.relu)# Activation function is ReLU
    #pooling layer #1
    #Pooling filter has zie 2X2 and stride 2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    #size is reduced by 50%. Shape of output is [batch_size, 14, 14, 32]

    #convolutional layer #2
    conv2 =tf.layers.conv2d(
        inputs = pool1,
        filters=64,         #Number of filters to apply
        kernel_size=[5, 5], #Filter dimensions [width, height]
        padding="same",     #Padding of 0 to preserce the dimensions of pooled layer
        activation= tf.nn.relu)

    #pooling layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    #There is again 50% reduction in dimensions of conv2. [batch_size, 7, 7, 64]

    #dense layer 

    #reshape pool2 to 2d [batch_size, features]
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    
    #A layer of 1024 neurons and RelU function to perform the classification
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    #Dropout regularization is added to reduce test errors
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode==tf.estimator.ModeKeys.TRAIN)
    #Output of the dropout layer is [batch_size, 1024]

    #logits layer
    #Dense Layer with 10 neurons for each class (0-9) with linear activation
    logits= tf.layers.dense(inputs=dropout, units=10)
    #Output shape [batch_size, 10]

    predictions ={
    #generate predictions for predict adn eval mode)
    "classes": tf.argmax(input=logits, axis=1),     #To find the greatest value
    "probabilities":tf.nn.softmax(logits, name="softmax_tensor") #To derive probabilities from logits 
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #converts labels in prediction to an 1X 10 array to match the output
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth=10)
    #calculate loss 
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    #configure the training Op
    #Model to optimize the loss using the learning rate =0.001 and stochastic gradietn descent
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op =optimizer.minimize(
            loss =loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op)

    eval_metric_ops={
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions= predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss = loss, eval_metric_ops=eval_metric_ops)

#Main function
def main(unused_argv):
    #loading training and eval data
    mnist= tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images #retruns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images #returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    #set up logging for predictions
    #tensors_to_log={"probabilities":"softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors=tensors_to_log, every_n_iter=500)


    #TRAIN the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=100,
        num_epochs= None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000)

    #evalulate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn= eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
