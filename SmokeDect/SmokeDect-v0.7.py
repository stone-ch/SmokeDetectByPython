# *^_^* coding:utf-8 *^_^*

from __future__ import print_function
import cv2
import numpy as np
import tensorflow as tf
import os
import random
# from matplotlib import pyplot as plt

__author__ = 'stone'
__date__ = '16-12-19'

DEBUG = False
AVERAGE_S_THRESHOLD = 70
HSV_V_BLOCK_COUNT = 50
CANDIDATE_BLOCK_SIZE = 20

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    print(y_pre)
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

    # tensorflow Variables
    xs = tf.placeholder(tf.float32, [None, 24*32])  # 32x24
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 24, 32, 1])

    # conv1 layer #
    W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 32x24x32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 16x12x32

    # conv2 layer #
    W_conv2 = weight_variable([5, 5, 32, 96])  # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([96])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 16x12x64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 8x6x64

    W_conv3 = weight_variable([5, 5, 96, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)    # output size 7x7x128

    # fc1 layer #
    W_fc1 = weight_variable([3*4*128, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool3_flat = tf.reshape(h_pool3, [-1, 3*4*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fc2 layer #
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "saver")
    sess.run(tf.initialize_all_variables())


    cap = cv2.VideoCapture(
        "../medias/myVideo/640x480/smoke1.avi")
    ret, start_frame = cap.read()
    start_gray_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        detectShadows=False
    )
    height, width = start_frame.shape[:2]
    frame_count = 0

    # save all blocks of the frame in HSV color space
    HSV_V_all_block = []
    while 1:
        ret, frame = cap.read()
        if frame is None:
            print("The End!")
            break
        smooth_kernel = np.ones((5, 5), np.float32)/25
        smooth_frame = cv2.filter2D(frame, -1, smooth_kernel)
        
        gray_frame = cv2.cvtColor(smooth_frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(smooth_frame, cv2.COLOR_BGR2HSV_FULL)
        if DEBUG:
            cv2.imshow("gray_frame", gray_frame)
            cv2.imshow("hsv_frame", hsv_frame)

        # GMM
        fgmask = fgbg.apply(gray_frame)
        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel2)
        fgmask = cv2.dilate(fgmask, kernel1)
        ret, fgmask_bin = cv2.threshold(fgmask, 0, 1, cv2.THRESH_BINARY)
        if DEBUG:
            ret, fgmask_bin_show = cv2.threshold(
                fgmask,
                0,
                255,
                cv2.THRESH_BINARY
            )
            cv2.imshow("fgmask_bin", fgmask_bin_show)

        block_width = width/CANDIDATE_BLOCK_SIZE
        block_height = height/CANDIDATE_BLOCK_SIZE
        HSV_V_each_block = []
        HSV_V_50_block = np.array(0)
        for m in range(0, width, block_width):
            for n in range(0, height, block_height):
                fgmask_clip = fgmask_bin[n:(block_height+n), m:(block_width+m)]
                candidate_clip = hsv_frame[n:(block_height+n), m:(block_width+m)]

                # store V of each frames
                HSV_V_each_block.append(np.average(candidate_clip[:, :, 2]))

                # find the move clips
                if fgmask_clip.any():
                    if DEBUG:
                        cv2.rectangle(frame, (m, n), (m+block_width, n+block_height), (255, 0, 0))

                    # average of S
                    candidate_clip_S = candidate_clip[:, :, 1]
                    average_S = np.average(candidate_clip_S)

                    # average of V
                    candidate_clip_V = candidate_clip[:, :, 2]
                    average_V = np.average(candidate_clip_V)

                    # if average of S lower than threshold it maybe smoke area
                    if (average_S < AVERAGE_S_THRESHOLD):
                        if DEBUG:
                            cv2.rectangle(frame, (m, n), (m+block_width, n+block_height), (0, 255, 0))

                        # the value of V in the smoke area is higher
                        HSV_V_all_block_ndarray = np.array(HSV_V_all_block)
                        if (frame_count > HSV_V_BLOCK_COUNT - 1):
                            HSV_V_50_block = HSV_V_all_block_ndarray[:, m/20]
                        elif (frame_count > 0):
                            HSV_V_50_block = HSV_V_all_block_ndarray[:frame_count, m/20]

                        if (np.average(HSV_V_50_block) - average_V > 0):
                            cv2.rectangle(frame, (m, n), (m+block_width, n+block_height), (0, 0, 255))
                            candidate_block = frame[n:(n+block_height), m:(m+block_width)]
                            candidate_block2 = cv2.cvtColor(candidate_block, cv2.COLOR_BGR2GRAY)
                            ret, candidate_block2 = cv2.threshold(candidate_block2, 125, 1, cv2.THRESH_BINARY)
                            candidate_block2_flat = np.reshape(candidate_block2, (1, -1))
                            # print("{},{},{}".format(frame_count, n, m))
                            # cv2.waitKey(10)
                            try:
                                result = sess.run(prediction, feed_dict={xs: candidate_block2_flat, keep_prob:1})
                                # print("CNN")
                                if result[0][0] > result[0][1]:
                                    print("find smoke at: frame{}({},{})".format(frame_count, n, m))
                                    cv2.rectangle(frame, (m, n), (m+block_width, n+block_height), (255, 255, 255))
                            except:
                                print("Bug{}".format(frame_count))
                                
                            # cv2.waitKey(0)

        # cv2.imshow("fgmask", fgmask)
        cv2.imshow("frame", frame)

        # store V of 50 frames before current frame
        if frame_count > HSV_V_BLOCK_COUNT - 1:
            HSV_V_all_block.pop(0)
            HSV_V_all_block.append(HSV_V_each_block)
            # print(HSV_V_all_block)
        else:
            HSV_V_all_block.append(HSV_V_each_block)

        frame_count += 1

        if (cv2.waitKey(10) & 0xFF) == 27:
            print("ESC")
            break

    cap.release()
    cv2.destroyAllWindows()
