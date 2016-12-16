# *^_^* coding:utf-8 *^_^*
"""
把图片统一输出为某一尺寸，或者原图片的n倍.
也可以截取图片中的某一区域。
"""
from __future__ import print_function

__author__ = 'stone'
__date__ = '16-7-14'

import cv2
import os

ZOOM_TIME = 1
START_X = 0
START_Y = 0
WIDTH = 55
HEIGHT = 60
IMG_PATH = '/home/st/Code/FlameSmokeDetect/medias/PictureForCNN/smoke_train' #  'dir/exampledir'
IMG_SAVE_PATH = '/home/st/Code/FlameSmokeDetect/medias/PictureForCNN/smoke_train28x28' #  'dir2/exampledir'


def zoom_down(imgs, time=None, size=None):
    """
    zoom down/up the image
    time:图片缩小的倍数
    size:要输出的图片的尺寸:size（width, height）
    """
    if time is not None:
        h, w, r = imgs.shape  # h:height w:width r:ret
        small_imgs = cv2.resize(img, (w / time, h / time), interpolation=cv2.INTER_CUBIC)
    elif size is not None:
        small_imgs = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return small_imgs


if __name__ == '__main__':
    for parent_dir, dirnames, files in os.walk(IMG_PATH):
        for img_name in files:
            img_path = parent_dir+"/"+img_name
            img_save_path = IMG_SAVE_PATH+"/"+img_name
            img = cv2.imread(img_path)

            # 用于缩放图片
            zoom_img = zoom_down(img, size=(28, 28))
            cv2.imwrite(img_save_path, zoom_img)
            cv2.imshow('img', zoom_img)

            """
            # 用于截取某一区域
            new_img = img[START_Y:START_Y+HEIGHT, START_X:START_X+WIDTH]
            # new_img = img[100:120, 100:150]

            cv2.imwrite(img_save_path, new_img)
            cv2.imshow('img', new_img)
            """

            if cv2.waitKey(20) & 0xFF == 27:
                break
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
