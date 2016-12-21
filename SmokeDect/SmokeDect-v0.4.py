# *^_^* coding:utf-8 *^_^*

from __future__ import print_function
import cv2
import numpy as np
# from matplotlib import pyplot as plt

__author__ = 'stone'
__date__ = '16-7-1'

DEBUG = True
AVERAGE_S_THRESHOLD = 70
HSV_V_BLOCK_COUNT = 50
CANDIDATE_BLOCK_SIZE = 20

if __name__ == "__main__":
    cap = cv2.VideoCapture(
        "../medias/myVideo/640x480/smoke5.avi")
    ret, start_frame = cap.read()
    start_gray_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        detectShadows=False
    )
    height, width = start_frame.shape[:2]
    frame_count = 0

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

                        if (np.average(HSV_V_50_block) - average_V < 5):
                            cv2.rectangle(frame, (m, n), (m+block_width, n+block_height), (0, 0, 255))
                            

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
