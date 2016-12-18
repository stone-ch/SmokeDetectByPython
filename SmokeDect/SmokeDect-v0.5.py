# *^_^* coding:utf-8 *^_^*
"""
Please note, this code is only for python 3+.
If you are using python 2+, please modify the code accordingly.

Classify smoke images by CNN:
    1. Only binary images.
    2. Input images 28*28*1 --> Level 1: output 28*28*32 --> pooling 1: 14*14*32
       --> Level 2: output 14*14*64 --> pooling 2: output 7*7*64 --> ANN: result
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import os
import random

__author__ = 'stone'
__date__ = '16-11-19'


def load_images(path):
    img_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            img = cv2.imread(path + filename, 0)
            ret, img2 = cv2.threshold(img, 125, 1, cv2.THRESH_BINARY)
            #cv2.imshow("img", img)
            img_flat = np.reshape(img2, (1, -1))
            img_list.append(img_flat)
    return img_list


# number 1 to 10 data


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    #print(y_pre)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    # print(x_image.shape)  # [n_samples, 28,28,1]
    #print(type(x_image))

    # conv1 layer #
    W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

    # conv2 layer #
    W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

    # fc1 layer #
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fc2 layer #
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.Session()
    # important step
    sess.run(tf.initialize_all_variables())
    train_smoke_images = load_images("../medias/PictureForCNN/smoke_train28x28/")
    train_none_smoke_images = load_images("../medias/PictureForCNN/none_smoke_train28x28/")
    test_smoke_images = load_images("../medias/PictureForCNN/smoke_test28x28/")
    test_none_smoke_images = load_images("../medias/PictureForCNN/none_smoke_test28x28/")
    total_train_images_list = []
    total_train_labels_list = []
    total_test_images_list = []
    total_test_labels_list = []
    for i in range(len(train_smoke_images)):
        total_train_images_list.extend(np.array(train_smoke_images[i], dtype=np.float32))
        total_train_labels_list.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        total_train_images_list.extend(np.array(train_none_smoke_images[i], dtype=np.float32))
        total_train_labels_list.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(len(test_smoke_images)):
        total_test_images_list.extend(test_smoke_images[i])
        total_test_labels_list.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        total_test_images_list.extend(test_none_smoke_images[i])
        total_test_labels_list.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    total_train_images = np.array(total_train_images_list, dtype=np.float32)
    total_train_labels = np.array(total_train_labels_list, dtype=np.float32)
    total_test_images = np.array(total_test_images_list, dtype=np.float32)
    total_test_labels = np.array(total_test_labels_list, dtype=np.float32)
    for index in range(100):
        # batch_xs, batch_ys = mnist.train.next_batch(100)
        i = random.randint(0, 130)
        batch_xs = total_train_images[i:i+50]
        batch_ys = total_train_labels[i:i+50]
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 5 == 0:
            #print(i, end='')
            print(compute_accuracy(
                total_test_images, total_test_labels))

    #batch_xs = total_train_images[i:i+10]
    #batch_ys = total_train_labels[i:i+10]
    #print(type(batch_xs))
    #print(np.shape(batch_xs))
    #print(batch_xs)
    #print(type(batch_ys))
    #print(np.shape(batch_ys))
    #print(batch_ys)
    #print(type(total_test_images))
    #print(np.shape(total_test_images))
    #print(total_test_images)
    #print(type(total_test_labels))
    #print(np.shape(total_test_labels))
    #print(total_test_labels)
