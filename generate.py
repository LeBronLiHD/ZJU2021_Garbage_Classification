# -*- coding: utf-8 -*-

import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from load_image import load_data
import parameter


def crop_circle(image):
    img_crop = image
    cen = image.shape[0]/2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if math.sqrt((i - cen)**2 + (j - cen)**2) > cen:
                img_crop[i][j][0] = 0
                img_crop[i][j][1] = 0
                img_crop[i][j][2] = 0
    return img_crop


def rotate_save(images, path):
    for i in range(len(images)):
        image = images[i]
        for r in range(int(parameter.GEN_RATE)):
            img_ro = image.rotate(r * (360/parameter.GEN_RATE))
            save_path = path + str(i) + "_" + str(r) + ".jpg"
            img_cv = crop_circle(np.array(img_ro))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path, img_cv)
            print(save_path)


def generate_date(train_images, train_labels, val_images, val_labels):
    for i in range(len(train_images)):
        images = train_images[i]
        label = train_labels[i][0]
        path = parameter.GEN_TRAIN
        path += "/" + parameter.INVERTED[label]
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        rotate_save(images, path=path + "/")
    for i in range(len(val_images)):
        images = val_images[i]
        label = val_labels[i][0]
        path = parameter.GEN_VAL
        path += "/" + parameter.INVERTED[label]
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        rotate_save(images, path=path + "/")


if __name__ == "__main__":
    all_train_images, all_train_labels = load_data(parameter.ORI_TRAIN)
    all_val_images, all_val_labels = load_data(parameter.ORI_VAL)
    generate_date(all_train_images, all_train_labels, all_val_images, all_val_labels)
