"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 13 October, 2017 @ 3:40 PM.
  Copyright © 2017. Victor. All rights reserved.
"""

# coding: utf-8

# import necessary dependencies and files
import os
import sys

import tensorflow as tf
import numpy as np

from models.config import DATASET_PATH, SAVED_FEATURES
from models.features import Features


# Load in the datasets
features = Features(data_dir=DATASET_PATH)
if os.path.isfile(SAVED_FEATURES):
    datasets = np.load(SAVED_FEATURES)
else:
    datasets = features.create(save_file=SAVED_FEATURES)

# Split into training and testing set
X_train, y_train, X_test, y_test = features.train_test_split(datasets)
print('Length of training set: {:,}'.format(len(y_train)))
print('Length of testing set:  {:,}'.format(len(y_test)))

# Define Hyperparameters
# Image & labels
image_size = features.image_size
image_channel = 3
image_shape = (image_size, image_size, image_channel)
image_shape_flat = image_size * image_size * image_channel
num_classes = len(features.classes)

# Network
filter_size = 5
hidden1_channels = 8
hidden2_channels = 16
hidden3_channels = 32
hidden4_channels = 64
hidden5_channels = 128
fully_connected_1 = 512
fully_connected_2 = 256

# Training
learning_rate = 1e-3
dropout = 0.8
iterations = 0
batch_size = 25


# Helper functions for `weights`, `biases`, `conv2d`, & `max_pool`
# Weight initialization
def weight(shape):
    initial = tf.truncated_normal(shape=shape, stddev=1.0/np.sqrt(shape[-2]))
    return tf.Variable(initial)


# Bias in initialization
def bias(length):
    initial = tf.zeros(shape=[length])
    return tf.Variable(initial)


# Convolutional operation
def conv2d(X, W):
    return tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')


# Max pooling operation
def max_pool(X):
    return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Flatten layer
def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = np.array(layer_shape[1:4], dtype=int).prod()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


# Placeholder Variables
X = tf.placeholder(tf.float32, [None, image_size, image_size, image_channel], name='X_input')
y = tf.placeholder(tf.float32, [None, num_classes], name='Y_input')
keep_prob = tf.placeholder(tf.float32)
y_true = tf.argmax(y, axis=1)


# Building the network
def convnet(train=False):
    with tf.name_scope('conv_layer1'):
        W_hidden1 = weight(shape=[filter_size, filter_size, image_channel, hidden1_channels])
        b_hidden1 = bias(length=hidden1_channels)
        h_conv1 = tf.nn.relu(conv2d(X, W_hidden1) + b_hidden1)
        h_pool1 = max_pool(h_conv1)
    with tf.name_scope('conv_layer2'):
        W_hidden2 = weight(shape=[filter_size, filter_size, hidden1_channels, hidden2_channels])
        b_hidden2 = bias(length=hidden2_channels)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_hidden2) + b_hidden2)
        h_pool2 = max_pool(h_conv2)
    with tf.name_scope('conv_layer3'):
        W_hidden3 = weight(shape=[filter_size, filter_size, hidden2_channels, hidden3_channels])
        b_hidden3 = bias(length=hidden3_channels)
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_hidden3) + b_hidden3)
        h_pool3 = max_pool(h_conv3)
    with tf.name_scope('conv_layer4'):
        W_hidden4 = weight(shape=[filter_size, filter_size, hidden3_channels, hidden4_channels])
        b_hidden4 = bias(length=hidden4_channels)
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_hidden4) + b_hidden4)
        h_pool4 = max_pool(h_conv4)
    with tf.name_scope('conv_layer5'):
        W_hidden5 = weight(shape=[filter_size, filter_size, hidden4_channels, hidden5_channels])
        b_hidden5 = bias(length=hidden5_channels)
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_hidden5) + b_hidden5)
        h_conv5_flat, num_features = flatten(h_conv5)
    with tf.name_scope('fully_connected1'):
        W_fc1 = weight(shape=[num_features, fully_connected_1])
        b_fc1 = bias(length=fully_connected_1)
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
        if train:
            h_fc1 = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
    with tf.name_scope('fully_connected2'):
        W_fc2 = weight(shape=[fully_connected_1, fully_connected_2])
        b_fc2 = bias(length=fully_connected_2)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        if train:
            h_drop = tf.nn.dropout(h_fc2, keep_prob=keep_prob)
    with tf.name_scope('output_layer'):
        W_out = weight(shape=[fully_connected_2, num_classes])
        b_out = bias(length=num_classes)
        y_pred = tf.matmul(h_drop, W_out) + b_out
        y_pred_norm = tf.nn.softmax(y_pred)
        y_pred_true = tf.argmax(y_pred_norm, axis=1)
    return y_pred, y_pred_norm, y_pred_true


# Run the convnet
y_pred, y_pred_norm, y_pred_true = convnet(train=True)
y_pred, y_pred_norm, y_pred_true


# Cost function, and optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y, name='xentropy')
cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Evaluating accuracy
correct = tf.equal(y_true, y_pred_true)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# Running tensorflow's `Session()`
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# feed_dict_train = {X: X_train, 
#                    y: y_train, 
#                    keep_prob:dropout}
feed_dict_test = {X: X_test, 
                  y: y_test, 
                  keep_prob: dropout}


# Optimize helper
def optimize(num_iter=1):
    global iterations
    for i in tqdm(range(num_iter)):
        start = 0
        while start < len(X_train):
            end = start + batch_size
            batch_X = X_train[start:end]
            batch_y = y_train[start:end]
            sess.run(optimizer, feed_dict={X:X_train,
                                           y:y_train,
                                           keep_prob:dropout})
            start += batch_size
        iterations += 1


# Accuracy helper
def eval_accuracy():
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    return acc


# Predict
def predict(img_path):
    img = feature.preprocess(img_path)
    _, y_pred_norm, y_pred_true = convnet()
    _pred = sess.run(y_pred_true, feed_dict={X: img})
    return features.classes[_pred]


if __name__ == '__main__':
    # Optimization
    optimize(10000) # to complete 10,000 iterations
    acc = eval_accuracy()
    print('Accuracy after {:,} iterations = {:.2%}'.format(iterations, acc))

    # Closing the tesorflow's `Session()`
    sess.close()

