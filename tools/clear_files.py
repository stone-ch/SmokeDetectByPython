# *^_^* coding:utf:8 *^_^*

from __future__ import print_function

__author__ = 'stone'
__date__ = '16-11-19'

import os

path = '/home/st/Code/FlameSmokeDetect/medias/PictureForCNN/'

for parent, dirnames, filenames in os.walk(path):
    for filename in filenames:
        if os.path.splitext(filename)[-1] == '.png':
            os.remove(path + filename)
