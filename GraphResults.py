
import itertools
from itertools import count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import Constants


def start():
    create_metric_graph()
    create_confusion_matrix()


def create_metric_graph():
    df_metric = pd.read_csv(Constants._RESULTS_FILENAME)

    print(df_metric)

    # plotting the bars
    colors = ['#f224aa', '#a52af7', '#24f2ef']
    bar_width = 0.8

    metric_plot = df_metric.plot.bar(rot = 0, width = bar_width, color = colors, alpha = 0.9)
    metric_plot.legend(loc = 'best')
    metric_plot.set_title('Model Performance')
    metric_plot.set_ylabel('Performance')
    metric_plot.set_xticklabels(Constants._LABELS)
    metric_plot.grid(linestyle = '-', linewidth = 0.4, axis = 'y')

    plt.show()


def create_confusion_matrix():
    y_true = np.load('./truey.npy')
    y_pred = np.load('./predy.npy')

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]


    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    title = 'Confusion matrix'

    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.YlGn)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation = 45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                horizontalalignment = "center",
                color = "white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()



