# -*- coding: utf-8 -*-

import os
import numpy
from PIL import Image
import parameter


def load_data(path):
    folders = os.listdir(path)
    all_images = []
    all_labels = []
    for folder in folders:
        folder_path = os.path.join(path, folder)
        cur_images = []
        cur_labels = []
        images = os.listdir(folder_path)
        for image in images:
            image_path = os.path.join(folder_path, image)
            img = Image.open(image_path)
            if img.size != (parameter.IMG_SIZE, parameter.IMG_SIZE):
                img = img.resize((parameter.IMG_SIZE, parameter.IMG_SIZE), resample=Image.BICUBIC)
            cur_images.append(img)
            cur_labels.append(parameter.INDEX[folder])
        all_images.append(cur_images)
        all_labels.append(cur_labels)
    print("read data done!")
    return all_images, all_labels


if __name__ == '__main__':
    all_train_images, all_train_labels = load_data(parameter.ORI_TRAIN)
    all_val_images, all_val_labels = load_data(parameter.ORI_VAL)
    print("all_train_images ->", len(all_train_images))
    print("all_val_images ->", len(all_val_images))
