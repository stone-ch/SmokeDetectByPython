# *^_^* coding:utf-8 *^_^*

"""
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import os

__author__ = 'stone'
__date__ = '2016-12-11'


def load_images(path):
    img_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            img = cv2.imread(path + filename, 0)
            ret, img2 = cv2.threshold(img, 125, 1, cv2.THRESH_BINARY)
            # cv2.imshow("img", img)
            img_flat = np.reshape(img2, (1, -1))
            img_list.append(img_flat)
    return img_list

if __name__ == "__main__":
    img = cv2.imread("../medias/smoke32_24.jpg")
    # print(np.shape(img))
    tf_img = tf.placeholder(tf.float32, [24, 32, 3])
    tf_img_4 = tf.reshape(tf_img, [-1, 24, 32, 3])
    tf_label = tf.placeholder(tf.float32, [2])
    keep_prob = tf.placeholder(tf.float32)

    # input [1, 24, 32, 3]
    # conv1 output:[1, 12, 16, 32]
    weight_variable1 = tf.Variable(
        tf.truncated_normal(
            [5, 5, 3, 32],
            stddev=0.1
        )
    )
    bias_variable1 = tf.Variable(
        tf.constant(
            0.1,
            shape=[32]
        )
    )
    conv1 = tf.nn.relu(
        tf.nn.conv2d(
            tf_img_4,
            weight_variable1,
            strides=[1, 1, 1, 1],
            padding="SAME"
        ) + bias_variable1
    )
    pooling1 = tf.nn.max_pool(
        conv1,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME"
    )

    # conv2 output:[1, 12, 16, 64]
    weight_variable2 = tf.Variable(
        tf.truncated_normal(
            [5, 5, 32, 64],
            stddev=0.1
        )
    )
    bias_variable2 = tf.Variable(
        tf.constant(
            0.1,
            shape=[64]
        )
    )
    conv2 = tf.nn.relu(
        tf.nn.conv2d
        (
            pooling1,
            weight_variable2,
            strides=[1, 1, 1, 1],
            padding="SAME"
        ) + bias_variable2
    )
    # pooling2 = tf.nn.max_pool(
    #     conv1,
    #     ksize=[1, 2, 2, 1],
    #     strides=[1, 2, 2, 1],
    #     padding="SAME"
    #     )

    # conv3 output:[1, 6, 8, 128]
    weight_variable3 = tf.Variable(
        tf.truncated_normal(
            [5, 5, 64, 128],
            stddev=0.1
        )
    )
    bias_variable3 = tf.Variable(
        tf.constant(
            0.1,
            shape=[128]
        )
    )
    conv3 = tf.nn.relu(
        tf.nn.conv2d(
            conv2,
            weight_variable3,
            strides=[1, 1, 1, 1],
            padding="SAME"
        ) + bias_variable3
    )
    pooling3 = tf.nn.max_pool(
        conv3,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME"
    )

    # conv4 output:[1, 6, 8, 256]
    weight_variable4 = tf.Variable(
        tf.truncated_normal(
            [5, 5, 128, 256],
            stddev=0.1
        )
    )
    bias_variable4 = tf.Variable(
        tf.constant(
            0.1,
            shape=[256]
        )
    )
    conv4 = tf.nn.relu(
        tf.nn.conv2d(
            pooling3,
            weight_variable4,
            strides=[1, 1, 1, 1],
            padding="SAME"
        ) + bias_variable4
    )
    # pooling4 = tf.nn.max_pool(
    #     conv3,
    #     ksize=[1, 2, 2, 1],
    #     strides=[1, 2, 2, 1],
    #     padding="SAME"
    # )

    # conv5 output:[1, 3, 4, 256]
    weight_variable5 = tf.Variable(
        tf.truncated_normal(
            [3, 3, 256, 256],
            stddev=0.1
        )
    )
    bias_variable5 = tf.Variable(
        tf.constant(
            0.1,
            shape=[256]
        )
    )
    conv5 = tf.nn.relu(
        tf.nn.conv2d(
            conv4,
            weight_variable5,
            strides=[1, 1, 1, 1],
            padding="SAME"
        ) + bias_variable5
    )
    pooling5 = tf.nn.max_pool(
        conv5,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME"
    )

    # fc1 layer
    weight_variable6 = tf.Variable(
        tf.truncated_normal(
            [3*4*256, 1024],
            stddev=0.1
        )
    )
    bias_variable6 = tf.Variable(
        tf.constant(
            0.1,
            shape=[1024]
        )
    )
    pooling5_flat = tf.reshape(pooling5, [-1, 3*4*256])
    fc1 = tf.nn.relu(
        tf.matmul(
            pooling5_flat,
            weight_variable6
        ) + bias_variable6
    )
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # fc2 layer
    weight_variable7 = tf.Variable(
        tf.truncated_normal(
            [1024, 2],
            stddev=0.1
        )
    )
    bias_variable7 = tf.Variable(
        tf.constant(
            0.1,
            shape=[2]
        )
    )
    prediction = tf.nn.softmax(
        tf.matmul(
            fc1_drop, weight_variable7
        ) + bias_variable7
    )

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(
            tf_label * tf.log(prediction),
            reduction_indices=[1]
        )
    )

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # load images
    train_smoke_images = load_images(
        "../medias/PictureForCNN/32x24/smoke_train_32x24/"
    )
    train_none_smoke_images = load_images(
        "../medias/PictureForCNN/32x24/none_smoke_train_32x24/"
    )
    test_smoke_images = load_images(
        "../medias/PictureForCNN/32x24/smoke_test_32x24/"
    )
    test_none_smoke_images = load_images(
        "../medias/PictureForCNN/32x24/none_smoke_test_32x24/"
    )
    total_train_images_list = []
    total_train_labels_list = []
    total_test_images_list = []
    total_test_labels_list = []
    for i in range(len(train_smoke_images)):
        total_train_images_list.extend(
            np.array(train_smoke_images[i], dtype=np.float32)
        )
        total_train_labels_list.append(
            [1, 0]
        )
        total_train_images_list.extend(
            np.array(train_none_smoke_images[i], dtype=np.float32)
        )
        total_train_labels_list.append(
            [0, 1]
        )

    for i in range(len(test_smoke_images)):
        total_test_images_list.extend(test_smoke_images[i])
        total_test_labels_list.append([1, 0])
        total_test_images_list.extend(test_none_smoke_images[i])
        total_test_labels_list.append([0, 1])
    # shape = tf.shape(pooling5)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        p = sess.run(
            prediction,
            feed_dict={tf_img: img, tf_label: [1, 0], keep_prob: 0.5}
        )
        print(p)
    
        # print(sess.run(weight_variable))
    # print("img/*****************img***************")
    # print(img)
    # cv2.imshow("img", img)
    # if cv2.waitKey(0) & 0xFF == 27:
    #    cv2.destroyAllWindows()
