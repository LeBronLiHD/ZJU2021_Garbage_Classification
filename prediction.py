# -*- coding: utf-8 -*-

import os
import random
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import parameter
import numpy as np
import math
import load_image
import train_model
from tensorflow.keras.utils import to_categorical


def model_prediction(path, test, label):
    model_lists = os.listdir(path)
    model_lists = sorted(model_lists,
                         key=lambda files: os.path.getmtime(os.path.join(path, files)),
                         reverse=False)
    model_lists.remove("tb_results")

    model = None
    if model_lists is None:
        print("no model found!")
    else:
        model = load_model(os.path.join(path, model_lists[0]))
        print("model loaded")

    predictions = model.predict(test)
    score = [0, 0]
    for i in range(len(test)):
        if np.argmax(predictions[i]) == np.argmax(label[i]):
            score[0] += 1
        else:
            score[1] += 1

    right, error = score[0]/len(test), score[1]/len(test)
    print("correct rate ->", right)
    print("error rate   ->", error)
    show_image, show_label, show_pred = [], [], []
    for i in range(parameter.PRE_SHOW):
        index = round(random.random() * len(test))
        show_image.append(test[index])
        show_label.append(np.argmax(label[index]))
        show_pred.append(np.argmax(predictions[index]))

    fig = plt.figure(figsize=(10, 10))
    for i in range(parameter.PRE_SHOW):
        ax = fig.add_subplot(round(math.sqrt(parameter.PRE_SHOW)), round(math.sqrt(parameter.PRE_SHOW)), i + 1)
        pred_label = parameter.INVERTED[show_pred[i]]
        true_label = parameter.INVERTED[show_label[i]]
        ax.set_title("pred:%s   truth:%s" % (show_pred[i], show_label[i]))
        plt.imshow(show_image[i], interpolation='nearest')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    all_val_images, all_val_labels = load_image.load_gen_data(parameter.GEN_VAL)
    all_val_images = np.array(all_val_images)
    all_val_labels = np.array(all_val_labels)

    print("original ->")
    print(type(all_val_images), type(all_val_labels))
    print(all_val_images.shape, all_val_labels.shape)

    all_val_images, all_val_labels = \
        train_model.flatten(all_val_images), train_model.flatten(all_val_labels)
    all_val_labels = to_categorical(all_val_labels, parameter.CLASS_NUM)
    shu_index = np.arange(all_val_images.shape[0])
    np.random.shuffle(shu_index)
    all_val_images = all_val_images[shu_index, :, :, :]
    all_val_labels = all_val_labels[shu_index, :]

    print("after preprocess ->")
    print(type(all_val_images), type(all_val_labels))
    print(all_val_images.shape, all_val_labels.shape)

    model_prediction(parameter.MODEL, all_val_images, all_val_labels)
