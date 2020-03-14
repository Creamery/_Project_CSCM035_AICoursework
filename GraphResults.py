
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def start():

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



