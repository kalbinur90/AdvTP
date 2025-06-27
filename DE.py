import os

import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import random
import math
# from detect import detection
# import pandas as pd


def img_light_pattern(img, path_adv, POP):

    a = POP.shape[0]
    # print('a = ', a)
    r, g, b = POP[0], POP[1], POP[2]
    pts = np.array([(POP[3], POP[4]), (POP[5], POP[6]), (POP[7], POP[8])], np.int32)

    if (a-3)/2 == 3:
        pts = np.array([(POP[3], POP[4]), (POP[5], POP[6]), (POP[7], POP[8])], np.int32)
    if (a-3)/2 == 4:
        pts = np.array([(POP[3], POP[4]), (POP[5], POP[6]), (POP[7], POP[8]), (POP[9], POP[10])], np.int32)
    if (a-3)/2 == 5:
        pts = np.array([(POP[3], POP[4]), (POP[5], POP[6]), (POP[7], POP[8]), (POP[9], POP[10]), (POP[11], POP[12])], np.int32)
    if (a-3)/2 == 6:
        pts = np.array([(POP[3], POP[4]), (POP[5], POP[6]), (POP[7], POP[8]), (POP[9], POP[10]), (POP[11], POP[12]), (POP[13], POP[14])], np.int32)
    if (a-3)/2 == 7:
        pts = np.array([(POP[3], POP[4]), (POP[5], POP[6]), (POP[7], POP[8]), (POP[9], POP[10]), (POP[11], POP[12]), (POP[13], POP[14]), (POP[15], POP[16])], np.int32)
    if (a-3)/2 == 8:
        pts = np.array([(POP[3], POP[4]), (POP[5], POP[6]), (POP[7], POP[8]), (POP[9], POP[10]), (POP[11], POP[12]), (POP[13], POP[14]), (POP[15], POP[16]), (POP[17], POP[18])], np.int32)
    if (a-3)/2 == 9:
        pts = np.array([(POP[3], POP[4]), (POP[5], POP[6]), (POP[7], POP[8]), (POP[9], POP[10]), (POP[11], POP[12]), (POP[13], POP[14]), (POP[15], POP[16]), (POP[17], POP[18]), (POP[19], POP[20])], np.int32)



    cv2.fillPoly(img, [pts], (r, g, b))

    cv2.imwrite(path_adv, img)

def light_intensity_adjustment(img, cnt, I, path_adv):

    if cnt == 0:
        return img

    height, width, n = img.shape

    mask = {
        1: cv2.imread(path_adv),
    }

    mask[cnt] = cv2.resize(mask[cnt], (width, height), interpolation=cv2.INTER_CUBIC)

    new_img = cv2.addWeighted(img, (1 - I), mask[cnt], I, 0)

    return new_img




def img_camera_sticker_visible(img, path, R, POP, X1, X2):

    b = POP.shape[0]
    # print('b = ', b)
    for i in range(int(b/3)):
        lenth = int(R * (X2 - X1)) if int(R * (X2 - X1)) > 0 else 1
        thickness = int(lenth / 2) if int(lenth / 2) > 0 else 1  # 宽度
        x1, y1, angle = int(POP[3*i+0]), int(POP[3*i+1]), int(POP[3*i+2])
        x2, y2 = int(x1 + lenth * (math.sin(math.radians(angle)))), int(y1 + lenth * (math.cos(math.radians(angle))))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    cv2.imwrite(path, img)


def img_camera_sticker_pattern_infrared_random(img, path, R, patch_number, x1, y1, x2, y2):

    # print('R, patch_number, x1, y1, x2, y2, R * (x2 - x1) = ', R, patch_number, x1, y1, x2, y2, R * (x2 - x1))

    for i in range(patch_number):
        w = int(R * (x2 - x1)) if int(R * (x2 - x1)) > 0 else 1
        h = w

        x, y, angel = random.randint(x1, x2), random.randint(y1, y2), random.randint(0, 180)
        angelPi = (angel / 180) * math.pi

        X1 = x + (w / 2) * math.cos(angelPi) - (h / 2) * math.sin(angelPi)
        Y1 = y + (w / 2) * math.sin(angelPi) + (h / 2) * math.cos(angelPi)

        X2 = x + (w / 2) * math.cos(angelPi) + (h / 2) * math.sin(angelPi)
        Y2 = y + (w / 2) * math.sin(angelPi) - (h / 2) * math.cos(angelPi)

        X3 = x - (w / 2) * math.cos(angelPi) + (h / 2) * math.sin(angelPi)
        Y3 = y - (w / 2) * math.sin(angelPi) - (h / 2) * math.cos(angelPi)

        X4 = x - (w / 2) * math.cos(angelPi) - (h / 2) * math.sin(angelPi)
        Y4 = y - (w / 2) * math.sin(angelPi) + (h / 2) * math.cos(angelPi)

        pts = np.array([(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)], np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 0))

    cv2.imwrite(path, img)



def img_camera_sticker_pattern_visible(img, path, R, POP, X1, X2):

    b = POP.shape[0]
    # print('b = ', b)
    for i in range(int(b/3)):
        lenth = int(R * (X2 - X1)) if int(R * (X2 - X1)) > 0 else 1
        thickness = int(lenth / 2) if int(lenth / 2) > 0 else 1  # 宽度
        x1, y1, angle = int(POP[3*i+0]), int(POP[3*i+1]), int(POP[3*i+2])
        x2, y2 = int(x1 + lenth * (math.sin(math.radians(angle)))), int(y1 + lenth * (math.cos(math.radians(angle))))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    cv2.imwrite(path, img)



def initiation(POP_num, size, x1, y1, x2, y2):
    POP = np.zeros((POP_num, size))
    for i in range(0, POP_num):
        POP[i][0] = random.randint(0, 255)
        POP[i][1] = random.randint(0, 255)
        POP[i][2] = random.randint(0, 255)
        for j in range(3, size):
            if j % 2 == 1:
                POP[i][j] = random.randint(x1, x2)
            if j % 2 == 0:
                POP[i][j] = random.randint(y1, y2)

    return POP


def mutation(POP, Rm, X1, Y1, X2, Y2):
    a, b = POP.shape
    POP1 = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            POP1[i][j] = POP[i][j]
    x, y = 0, 0

    #变异
    for i in range(a):

        for tag in range(0, 100):
            x, y = random.randint(0, a-1), random.randint(0, a-1)
            if x != i and y != i and x != y :
                break
        # print('i, x, y = ', i, x, y)

        for j in range(b):
            POP[i][j] = POP1[i][j] + Rm * (POP1[x][j] - POP1[y][j])
    # print('POP = ', POP)

    #取整
    for i in range(a):
        for j in range(b):
            POP[i][j] = int(POP[i][j])
    # print('POP = ', POP)

    #Clip
    for i in range(0, a):

        for j in range(0, 3):
            if POP[i][j] < 0 or POP[i][j] > 255:
                POP[i][j] = random.randint(0, 255)
        for j in range(3, b):
            if j % 2 == 1:
                if POP[i][j] < X1 or POP[i][j] > X2:
                    POP[i][j] = random.randint(X1, X2)
            if j % 2 == 0:
                if POP[i][j] < Y1 or POP[i][j] > Y2:
                    POP[i][j] = random.randint(Y1, Y2)
    # print('POP = ', POP)

    return POP

def crossover(POP, POP_f, Rc):
    a, b = POP.shape
    for i in range(a):
        for j in range(b):
            tag = random.randint(1, 10)
            # print('i, j, tag = ', i, j, tag)
            if tag <= 10 * Rc:
                POP[i][j] = POP_f[i][j]
    return POP


def selection(POP, POP_f, POP_conf, POP_conf_f):
    a, b = POP.shape
    for i in range(a):
        if POP_conf[0][i] > POP_conf_f[0][i]:
            for j in range(b):
                POP[i][j] = POP_f[i][j]

    return POP

def rate(side_length):
    R = 0
    if side_length == 0:
        R = 0.06
    if side_length == 1:
        R = 0.08
    if side_length == 2:
        R = 0.1
    if side_length == 3:
        R = 0.12
    if side_length == 4:
        R = 0.14
    if side_length == 5:
        R = 0.16

    return R



# POP = initiation(5, 3, 0, 0, 5, 10)
# POP_f = initiation(5, 3, 0, 0, 5, 10)
# print('POP = ', POP)
# print('POP_f = ', POP_f)
# POP = crossover(POP, POP_f, 0.6)
# print('POP = ', POP)
# POP = mutation(POP, 0.5, 0, 0, 5, 10)
# print('POP = ', POP)





























