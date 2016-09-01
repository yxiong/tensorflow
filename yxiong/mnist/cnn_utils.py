#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Aug 30, 2016.

import tensorflow as tf

def inference(x, keep_prob):
    """Build a CNN for inference.

    Args:
      x: a placeholder node of shape [None, 784].
      keep_prob: a control node for dropout, a placeholder of type float.

    Returns:
      y: output node for the probability of each label, of shape [None, 10]"""
    # First convolutional layer.
    x_image = tf.reshape(x, [-1,28,28,1])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer (fully connected followed by softmax).
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y


def loss(y, y_):
    """Calculate the loss from the inference output and labels.

    Args:
      y: the output probabilities from inference()
      y_: the one-hot representation of labels, of shape [None, 10]

    Returns:
      loss: loss tensor of type float
    """
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1], name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
    return loss


def accuracy(y, y_):
    """Calculate the accuracy from the inference output and labels."""
    cross_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name="accuracy")
    accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32), name="accuracy_mean")
    return accuracy


def training(loss, accuracy, optimizer):
    tf.scalar_summary(loss.op.name, loss)
    tf.scalar_summary(accuracy.op.name, accuracy)
    global_step = tf.Variable(0, name = "global-step", trainable = False)
    return optimizer.minimize(loss)


################################################################
# Some convenience functions for building the graph.
################################################################

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")
