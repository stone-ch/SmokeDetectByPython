# coding:utf-8
"""
Create 3D scatter diagram of the images in different color spaces
"""

from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import os


def draw__RGB_figure(img_path):
    img = cv2.imread(img_path)

    _X = img[:, :, 0]
    _Y = img[:, :, 1]
    _Z = img[:, :, 2]

    X = np.reshape(_X, (1, -1))
    Y = np.reshape(_Y, (1, -1))
    Z = np.reshape(_Z, (1, -1))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(X, Y, Z)
    ax.set_xlabel("B-x")
    ax.set_ylabel("G-y")
    ax.set_zlabel("R-z")
    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)

    figure_path = img_path+"_BGR_figure.png"
    plt.savefig(figure_path, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    # plt.show()
    plt.close()


def draw_HSV_figure(img_path):
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    _X = hsv_img[:, :, 0]
    _Y = hsv_img[:, :, 1]
    _Z = hsv_img[:, :, 2]

    X = np.reshape(_X, (1, -1))
    Y = np.reshape(_Y, (1, -1))
    Z = np.reshape(_Z, (1, -1))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(X, Y, Z)
    ax.set_xlabel("H-x")
    ax.set_ylabel("S-y")
    ax.set_zlabel("V-z")
    ax.set_xlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_zlim3d(0, 255)

    figure_path = img_path+"_HSV_figure.png"
    plt.savefig(figure_path, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    # img_path_root example: "../medias/smoke_pic/test"
    img_path_root = "../medias/smoke_pic/test"    # the directory path
    # ergodic the directory
    figure_count = 0
    for parent_dir, dirnames, files in os.walk(img_path_root):
        for img_name in files:
            if img_name.split('.')[-1] == 'jpg':
                img_path = parent_dir+"/"+img_name
                print(img_path)
                figure_count += 1
                draw_HSV_figure(img_path)
    print("total: {}".format(figure_count))
    if figure_count == 0:
        print("ATTENTIOM: NO IMAGES OR WRONG DIRECTORY!!!!!!")
