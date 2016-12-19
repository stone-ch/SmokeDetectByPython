# *^_^* coding:utf-8 *^_^*

"""
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import os
import random

__author__ = 'stone'
__date__ = '2016-12-11'


def load_images(path):
    img_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            img = cv2.imread(path + filename, 0)
            ret, img2 = cv2.threshold(img, 125, 1, cv2.THRESH_BINARY)
            img_flat = np.reshape(img, (1, -1))
            # print(np.shape(img_flat))
            img_list.append(img_flat)
    return img_list


def compute_accuracy(v_xs, v_ys):
    print("accuracy")
    global prediction
    y_pre = sess.run(
        prediction,
        feed_dict={tf_img: v_xs, keep_prob: 1}
    )
    # print(y_pre)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(
        accuracy,
        feed_dict={tf_img: v_xs, tf_label: v_ys, keep_prob: 1}
    )
    return result

if __name__ == "__main__":
    # img = cv2.imread("../medias/smoke32_24.jpg")
    # tf_img = tf.placeholder(tf.float32, [24, 32, 3])
    tf_img = tf.placeholder(tf.float32, [None, 24*32])
    tf_img_4 = tf.reshape(tf_img, [-1, 24, 32, 1])
    tf_label = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    # input [1, 24, 32, 3]
    # conv1 output:[1, 12, 16, 96]
    weight_variable1 = tf.Variable(
        tf.truncated_normal(
            [5, 5, 1, 32],
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
    pooling2 = tf.nn.max_pool(     # if pooling output:[1, 6, 8, 128]
        conv2,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME"
    )

    # # conv3 output:[1, 6, 8, 128]
    # weight_variable3 = tf.Variable(
    #     tf.truncated_normal(
    #         [5, 5, 128, 256],
    #         stddev=0.1
    #     )
    # )
    # bias_variable3 = tf.Variable(
    #     tf.constant(
    #         0.1,
    #         shape=[256]
    #     )
    # )
    # conv3 = tf.nn.relu(
    #     tf.nn.conv2d(
    #         conv2,
    #         weight_variable3,
    #         strides=[1, 1, 1, 1],
    #         padding="SAME"
    #     ) + bias_variable3
    # )
    # pooling3 = tf.nn.max_pool(    # if pooling output:[1, 3, 4, 128]
    #     conv3,
    #     ksize=[1, 2, 2, 1],
    #     strides=[1, 2, 2, 1],
    #     padding="SAME"
    # )
    # 
    # # conv4 output:[1, 6, 8, 256]
    # weight_variable4 = tf.Variable(
    #     tf.truncated_normal(
    #         [3, 3, 256, 256],
    #         stddev=0.1
    #     )
    # )
    # bias_variable4 = tf.Variable(
    #     tf.constant(
    #         0.1,
    #         shape=[256]
    #     )
    # )
    # conv4 = tf.nn.relu(
    #     tf.nn.conv2d(
    #         pooling3,
    #         weight_variable4,
    #         strides=[1, 1, 1, 1],
    #         padding="SAME"
    #     ) + bias_variable4
    # )
    # # pooling4 = tf.nn.max_pool(
    # #     conv3,
    # #     ksize=[1, 2, 2, 1],
    # #     strides=[1, 2, 2, 1],
    # #     padding="SAME"
    # # )
    # 
    # # conv5 output:[1, 3, 4, 128]
    # weight_variable5 = tf.Variable(
    #     tf.truncated_normal(
    #         [3, 3, 256, 128],
    #         stddev=0.1
    #     )
    # )
    # bias_variable5 = tf.Variable(
    #     tf.constant(
    #         0.1,
    #         shape=[128]
    #     )
    # )
    # conv5 = tf.nn.relu(
    #     tf.nn.conv2d(
    #         conv4,
    #         weight_variable5,
    #         strides=[1, 1, 1, 1],
    #         padding="SAME"
    #     ) + bias_variable5
    # )
    # pooling5 = tf.nn.max_pool(
    #     conv5,
    #     ksize=[1, 2, 2, 1],
    #     strides=[1, 2, 2, 1],
    #     padding="SAME"
    # )

    # fc1 layer
    weight_variable6 = tf.Variable(
        tf.truncated_normal(
            [6*8*64, 1024],
            stddev=0.1
        )
    )
    bias_variable6 = tf.Variable(
        tf.constant(
            0.1,
            shape=[1024]
        )
    )
    pooling5_flat = tf.reshape(pooling2, [-1, 6*8*64])
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

    # cross_entropy = tf.reduce_mean(
    #     -tf.reduce_sum(
    #         tf_label * tf.log(prediction),
    #         reduction_indices=[1]
    #     )
    # )
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(
            tf_label * tf.log(prediction),
            1
        )
    )

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # load images
    train_smoke_images = []
    train_smoke_images = load_images(
        "../medias/PictureForCNN/test/smoke_train_32x24/"
    )
    train_none_smoke_images = load_images(
        "../medias/PictureForCNN/test/none_smoke_train_32x24/"
    )
    test_smoke_images = load_images(
        "../medias/PictureForCNN/test/smoke_test_32x24/"
    )
    test_none_smoke_images = load_images(
        "../medias/PictureForCNN/test/none_smoke_test_32x24/"
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

    total_train_images = np.array(total_train_images_list, dtype=np.float32)
    total_train_labels = np.array(total_train_labels_list, dtype=np.float32)
    total_test_images = np.array(total_test_images_list, dtype=np.float32)
    total_test_labels = np.array(total_test_labels_list, dtype=np.float32)
    # shape = tf.shape(pooling5)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for index in range(100):
            i = random.randint(0, 130)
            batch_images = total_train_images[i: i+50]
            batch_labels = total_train_labels[i: i+50]
            # print(sess.run(
            #           shape,
            #           feed_dict={tf_img: batch_images}
            #           )
            #       )
            # print(np.shape(batch_images))
            # print(np.shape(batch_labels))
            sess.run(train_step, feed_dict={tf_img: batch_images,tf_label: batch_labels,keep_prob: 0.5})
            # print(p)
            # p = sess.run(tf.shape(pooling5_flat), feed_dict={tf_img: batch_images,tf_label: batch_labels,keep_prob: 0.5})
            # print(p)
            if index % 10 == 0:
                print("{} ".format(index), end='')
                print(
                    compute_accuracy(
                        total_test_images, total_test_labels
                    )
                )

    # with tf.Session() as sess:
    #     sess.run(tf.initialize_all_variables())
    #     print(sess.run(tf.shape(pooling1), feed_dict={tf_img: img}))
        # print(sess.run(weight_variable))
    # print("img/*****************img***************")
    # print(img)
    # cv2.imshow("img", img)
    # if cv2.waitKey(0) & 0xFF == 27:
    #    cv2.destroyAllWindows()
