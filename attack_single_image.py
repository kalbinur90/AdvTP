import os

import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import random
import math
from detect_single_img import detect
from DE import img_light_pattern, initiation, mutation, crossover, selection, rate, light_intensity_adjustment
import pandas as pd


def img_light_pattern_sim(img, path_adv, POP):

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



    cv2.fillPoly(img, [pts], (0, 0, 255))

    cv2.imwrite(path_adv, img)


aa = 0


count = np.zeros((7, 5))
query = np.zeros((7, 5))
Rm = 0.5
Rc = 0.6


suc_id = 11

# path = r"C:\Users\a\PycharmProjects\pythonProject\KALI1\TT100kcoco\mnt\data0\home\zhoujingzhi\detection_dataset\TT100K-coco\images\test\5163.jpg"

Light = 3

for light_shape in range(Light, Light+1):#3-9
    for Intensity in range(5, 6, 2):#1-10-2

        dir = r'C:\Users\a\PycharmProjects\pythonProject\KALI1\TT100kcoco\mnt\data0\home\zhoujingzhi\detection_dataset\TT100K-coco\images\test'
        pic = os.listdir(dir)
        POP_num = 100  # 种群大小
        size = 3 + 2 * light_shape  # 每个个体有多少基因，r, g, b, x1, y1, x2, y2, x3, y3...
        Step = 10  # 迭代次数

        pic_number = 0
        for pic_id in range(0, 970):
            print('pic_id = ', pic_id)
            # pic_dir = dir + '/' + pic[pic_id]
            pic_dir = r'C:\Users\a\Desktop\1.jpg'
            # print(pic_dir)
            xmin, ymin, xmax, ymax, conf, shape = detect(pic_dir)


            if shape != 1:
                continue

            pic_number = pic_number + 1
            print('pic_id, pic_number', pic_id, pic_number)
            # print('xmin, ymin, xmax, ymax, conf, shape = ', xmin, ymin, xmax, ymax, conf, shape)


            POP = initiation(POP_num, size, xmin, ymin, xmax, ymax)
            POP_conf = np.ones((1, POP_num))
            # print('POP = ', POP)
            # print('POP_conf = ', POP_conf)

            POP_f = initiation(POP_num, size, xmin, ymin, xmax, ymax)
            POP_conf_f = np.ones((1, POP_num))

            # 获取初始父代置信度
            tag_break = 0
            for individual in range(0, POP_num):

                # print('int((Intensity-1)/2) = ', int((Intensity-1)/2))
                column = int((Intensity-1)/2)
                query[light_shape - 3][column] = query[light_shape - 3][column] + 1

                img = cv2.imread(pic_dir)
                path_adv = 'adv.jpg'

                print('父代初始化 pic_id, pic_number, individual = ', pic_id, pic_number, individual)
                print('suc_id, query = ', suc_id, query[light_shape - 3][column])

                img_light_pattern_sim(img, path_adv, POP[individual])

                cap = cv2.VideoCapture(pic_dir)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                i = 1
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                videoWriter = cv2.VideoWriter(path_adv, fourcc, fps, (width, height))
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        frame = light_intensity_adjustment(frame, i % 5, Intensity/10, path_adv)
                        # cv2.imshow('video', frame)
                        videoWriter.write(frame)
                        i += 1
                        c = cv2.waitKey(1)
                        if c == 27:
                            break
                    else:
                        break

                # img_show = plt.imread(path_adv)
                # plt.imshow(img_show)
                # plt.show()




                result_adv = detect(path_adv)
                # print('result_adv = ', result_adv)
                print('result_adv.shape = ', result_adv[5])
                # print('result_adv[0].shape = ', result_adv[0].shape)
                if result_adv[5] == 0:
                    adv_save = 'sim_samples/' + str(suc_id) + '.jpg'
                    suc_id = suc_id + 1
                    img_save = cv2.imread(path_adv)
                    cv2.imwrite(adv_save, img_save)
                    count[light_shape - 3][column] = count[light_shape - 3][column] + 1
                    tag_break = 1
                    break
                else:
                    POP_conf[0][individual] = result_adv[4]

                # print(result_adv[0][0][4])

            # print('POP = ', POP)
            # print('POP_conf = ', POP_conf)

            if tag_break == 1:
                continue

            for i in range(POP_num):
                for j in range(size):
                    POP_f[i][j] = POP[i][j]
            for i in range(POP_num):
                POP_conf_f[0][i] = POP_conf[0][i]







            # 开始DE攻击
            for step in range(Step):

                # print('POP = ', POP)
                POP = mutation(POP, Rm, xmin, ymin, xmax, ymax)
                # print('POP = ', POP)
                POP = crossover(POP, POP_f, Rc)
                # print('POP = ', POP)

                tag_break = 0

                for individual in range(0, POP_num):

                    column = int((Intensity - 1) / 2)
                    query[light_shape - 3][column] = query[light_shape - 3][column] + 1

                    img = cv2.imread(pic_dir)
                    path_adv = 'adv.jpg'

                    # print('individual = ', individual)

                    img_light_pattern_sim(img, path_adv, POP[individual])
                    # img_light_pattern(img, path_adv, rate(side_length), POP[individual], ymin, ymax)

                    cap = cv2.VideoCapture(pic_dir)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    i = 1
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    videoWriter = cv2.VideoWriter(path_adv, fourcc, fps, (width, height))
                    while (cap.isOpened()):
                        ret, frame = cap.read()
                        if ret:
                            frame = light_intensity_adjustment(frame, i % 5, Intensity / 10, path_adv)
                            # cv2.imshow('video', frame)
                            videoWriter.write(frame)
                            i += 1
                            c = cv2.waitKey(1)
                            if c == 27:
                                break
                        else:
                            break


                    result_adv = detect(path_adv)
                    print('result_adv.shape = ', result_adv[5])
                    if result_adv[5] == 0:
                        adv_save = 'sim_samples/' + str(suc_id) + '.jpg'
                        suc_id = suc_id + 1
                        img_save = cv2.imread(path_adv)
                        cv2.imwrite(adv_save, img_save)
                        count[light_shape - 3][column] = count[light_shape - 3][column] + 1
                        tag_break = 1
                        break
                    # print(result_adv[0][4])
                    POP_conf[0][individual] = result_adv[4]

                    print('light_shape, Intensity, pic_id, pic_number, step, individual = ', light_shape, Intensity, pic_id, pic_number, step, individual)
                    print('suc_id, query = ', suc_id, query[light_shape - 3][column])



                    # img_show = plt.imread(path_adv)
                    # plt.imshow(img_show)
                    # plt.show()

                print('count = ', count)
                print('query = ', query)
                if tag_break == 1:
                    break

                # print('POP = ', POP)
                # print('POP_f = ', POP_f)
                # print('POP_conf = ', POP_conf)
                # print('POP_conf_f = ', POP_conf_f)
                POP = selection(POP, POP_f, POP_conf, POP_conf_f)
                # print('POP = ', POP)





            # if pic_number == 1:
            #     break

        print('count = ', count)
        print('query = ', query)
        print('count/280 = ', count/280)#v5是280张
        print('query/280 = ', query/280)
        print('count/pic_number = ', count / pic_number)
        print('query/pic_number = ', query / pic_number)




