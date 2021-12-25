# -*- coding: utf-8 -*-

import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from load_image import load_data
import parameter
import random
import shutil
import sys


def five_index(cover, number):
    index = []
    for i in range(number):
        new = round(random.random() * cover)
        is_overlap = True
        while is_overlap:
            is_overlap = False
            for j in range(len(index)):
                if new == index[j]:
                    print("already selected, re-find a new one")
                    is_overlap = True
                    break
            if is_overlap is False:
                break
            new = round(random.random() * cover)
        index.append(new)
    return index


def delete_ori_val(path):
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        images = os.listdir(folder_path)
        for image in images:
            image_path = os.path.join(folder_path, image)
            os.remove(image_path)


def generate_val(train_path, val_path):
    train_folders = os.listdir(train_path)
    val_folders = os.listdir(val_path)
    for i in range(len(train_folders)):
        train_folder = train_folders[i]
        val_folder = val_folders[i]
        if train_folder != val_folder:
            print("train_folder != val_folder different folder name!")
            return
        train_folder_path = os.path.join(train_path, train_folder)
        val_folder_path = os.path.join(val_path, val_folder)
        images = os.listdir(train_folder_path)
        select = five_index(len(images) - 1, 5)
        images_select = []
        for j in range(5):
            print("select ->", select)
            images_select.append(images[select[j]])
        for j in range(5):
            train_image = os.path.join(train_folder_path, images_select[j])
            val_image = os.path.join(val_folder_path, images_select[j])
            try:
                shutil.move(train_image, val_folder_path)
            except IOError as error:
                print("Unable to move file. %s" % error)
            except:
                print("Unexpected error:", sys.exc_info())


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
    # # ONLY RUN ONE TIME!!!!
    # # COPY FROM data/ IF RUN MORE THAN ONCE!!!
    # delete_ori_val(parameter.ORI_VAL)
    # generate_val(parameter.ORI_TRAIN, parameter.ORI_VAL)
    all_train_images, all_train_labels = load_data(parameter.ORI_TRAIN)
    all_val_images, all_val_labels = load_data(parameter.ORI_VAL)
    generate_date(all_train_images, all_train_labels, all_val_images, all_val_labels)
