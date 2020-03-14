
# load json and create model
from __future__ import division

from keras.models import model_from_json
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

import ExportCsv


def start():

    json_file = open('fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("fer.h5")
    print("Loaded model from disk")

    truey = []
    predy = []

    x = np.load('./modXtest.npy')
    y = np.load('./modytest.npy')

    yhat = loaded_model.predict(x)
    yh = yhat.tolist()
    yt = y.tolist()
    count = 0
    for i in range(len(y)):
        yy = max(yh[i])
        yyt = max(yt[i])
        predy.append(yh[i].index(yy))
        truey.append(yt[i].index(yyt))

        if(yh[i].index(yy) == yt[i].index(yyt)):
            count += 1

    total = len(y)
    acc = (count/total) * 100
    results = sklearn.metrics.precision_recall_fscore_support(truey, predy)
    # prec = average_precision_score(truey, predy)

    # save values for confusion matrix
    np.save('truey', truey)
    np.save('predy', predy)
    print("Predicted and true label values saved")
    print("Correct : " + str(count))
    print("Total : " + str(total))
    print("Accuracy on test set :" + str(acc) + "%")

    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print("")
    print("--------------------------------")
    print("Precision and Recall per Emotion")
    print("--------------------------------")

    header = labels.copy()
    header.insert(0, "Metric")

    len_labels = len(labels)
    precision_row = []
    recall_row = []
    fscore_row = []


    for i in range(len_labels):
        print(labels[i])

        precision = results[0][i]
        round_precision = round(precision * 100, 3)
        precision_row.append(round_precision)

        recall = results[1][i]
        round_recall = round(recall * 100, 3)
        recall_row.append(round_recall)

        fscore = 2 * ((precision * recall)/(precision + recall))
        round_fscore = round(fscore * 100, 3)
        fscore_row.append(round_fscore)

        print("Precision : " + str(round_precision) + " %")
        print("Recall : " + str(round_recall) + " %")
        print("F-score : " + str(round_fscore) + " %")
        print("")

    precision_row.insert(0, "Precision")
    recall_row.insert(0, "Recall")
    fscore_row.insert(0, "F-Score")

    result_table = np.vstack([header, precision_row, recall_row, fscore_row])
    result_table = result_table.transpose()
    ExportCsv.save(result_table)
