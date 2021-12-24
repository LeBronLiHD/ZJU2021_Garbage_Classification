# -*- coding: utf-8 -*-
import os
import random
from keras.models import load_model
import matplotlib.pyplot as plt
import parameter


def plot_load_and_model_prediction(path, test, label):
    model_lists = os.listdir(path)
    model_lists = sorted(model_lists,
                         key=lambda files: os.path.getmtime(os.path.join(path, files)),
                         reverse=False)
    model = None
    if model_lists is None:
        print("no model found!")
    else:
        model = load_model(os.path.join(path, model_lists[0]))
        print("model loaded")

    images = []
    answer = []
    for i in range(len(test)):
        for j in range(len(test[i])):
            images.append(test[i][j])
            answer.append(label[i][j])

    predictions = model.predict(images)
    score = [0, 0]
    for i in range(len(images)):
        if predictions[i] == answer[i]:
            score[0] += 1
        else:
            score[1] += 1

    right, error = score[0]/len(images), score[1]/len(images)
    print("correct rate ->", right)
    print("error rate   ->", error)
    show_image, show_label = [], []
    for i in range(parameter.PRE_SHOW):
        index = round(random.random() * len(images))
        show_image.append(images[index])
        show_label.append((label[index]))

    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.title(
                'pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
        plt.imshow(test_x[i])
