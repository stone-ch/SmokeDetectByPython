# coding:utf-8
"""
Create 2D scatter diagram of the images in different color spaces
"""

from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

__author__ = 'stone'
__date__ = '16-7-14'


def draw_BGR_figure(img_path):
    figure_path = img_path+"_RGB_2D_figure.png"
    img = cv2.imread(img_path)

    _img_b = img[:, :, 0]
    _img_g = img[:, :, 1]
    _img_r = img[:, :, 2]

    img_b = np.reshape(_img_b, (1, -1))[0]
    img_g = np.reshape(_img_g, (1, -1))[0]
    img_r = np.reshape(_img_r, (1, -1))[0]

    plt.axis([0, simple_count-1, 0, 300])
    simple_index = np.shape(img)[0]*np.shape(img)[1]/simple_count
    plt.plot(img_b[::simple_index], color='b', label='B')
    plt.plot(img_g[::simple_index], color='g', label='G')
    plt.plot(img_r[::simple_index], color='r', label='R')

    plt.savefig(figure_path, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.close()


def draw_HSV_figure(img_path):
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    _img_h = hsv_img[:, :, 0]
    _img_s = hsv_img[:, :, 1]
    _img_v = hsv_img[:, :, 2]

    img_h = np.reshape(_img_h, (1, -1))[0]
    img_s = np.reshape(_img_s, (1, -1))[0]
    img_v = np.reshape(_img_v, (1, -1))[0]

    # S-V
    ax = plt.gca()
    ax.set_xlabel("x->S")
    ax.set_ylabel("y->V")
    plt.axis([0, 255, 0, 255])
    plt.scatter(img_s, img_v)
    figure_path = img_path+"_SV_2D_figure.png"
    plt.savefig(figure_path, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.close()

    # S-H
    ax = plt.gca()
    ax.set_xlabel("x->S")
    ax.set_ylabel("y->H")
    plt.axis([0, 255, 0, 255])
    plt.scatter(img_s, img_h)
    figure_path = img_path+"_SH_2D_figure.png"
    plt.savefig(figure_path, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.close()

    # V-H
    ax = plt.gca()
    ax.set_xlabel("x->V")
    ax.set_ylabel("y->H")
    plt.axis([0, 255, 0, 255])
    plt.scatter(img_v, img_h)
    figure_path = img_path+"_VH_2D_figure.png"
    plt.savefig(figure_path, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.close()

if __name__ == "__main__":
    simple_count = 101
    # img_path_root example: "../medias/smoke_pic/test"
    img_path_root = "../medias/smoke_pic/smoke_clip"    # the directory path
    # ergodic the directory
    figure_count = 0
    for parent_dir, dirnames, files in os.walk(img_path_root):
        for img_name in files:
            if img_name.split('.')[1] == 'png':
                img_path = parent_dir+"/"+img_name
                print(img_path)
                figure_count += 1
                draw_BGR_figure(img_path)
    print("total: {}".format(figure_count))
    if figure_count == 0:
        print("ATTENTIOM: NO IMAGES OR WRONG DIRECTORY!!!!!!")

