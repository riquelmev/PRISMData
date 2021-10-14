import pickle
import keras
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
file = "/home/vicente/storage/data/modelpredictions50msTest.pickle"
with open(file, 'rb') as f:
    y_pred_ohe = pickle.load(f)

#file = '/mount/storage/data/y_validation_2catagories50ms.pickle'
file = "/home/vicente/storage/data/modelactual50msTest.pickle"
with open(file, 'rb') as f:
    y_cat = pickle.load(f)
model = keras.models.load_model('/home/vicente/storage/2classmodelTest0.82')

print(y_pred_ohe)
print(y_cat)
loss = []
no_loss =[]
values = []
for i in range(1,30):
    y_pred_labels = []
    for value in y_pred_ohe:
        if float(value[1]) == 0 or float(value[0]) / float(value[1]) > i:
            y_pred_labels.append(0)
        else:
            y_pred_labels.append(1)
    y_pred_labels = np.asarray(y_pred_labels)
    matrix = metrics.confusion_matrix(y_cat, y_pred_labels)

    # matrix = metrics.confusion_matrix(y_val.argmax(axis=1), pred.argmax(axis=1))

    print(matrix)
    print(matrix[1][1]/(matrix[1][1]+matrix[1][0]))
    print(matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    loss_recall = matrix[1][1]/(matrix[1][1]+matrix[1][0])
    noloss_recall = matrix[0][0]/(matrix[0][0]+matrix[0][1])
    loss.append(loss_recall)
    no_loss.append(noloss_recall)

print(classification_report(y_cat, y_pred_labels))

y_pred_labels = np.argmax(y_pred_ohe, axis=1)
matrix = metrics.confusion_matrix(y_cat, y_pred_labels)
print(matrix)
print(classification_report(y_cat, y_pred_labels))

print(loss)
print(no_loss)

with open('newAccuracyLoss.txt', 'w') as f:
    for i in range(len(loss)):
        f.write(str(loss[i]) + "\n")
with open('newAccuracyNoLoss.txt', 'w') as f:
    for i in range(len(no_loss)):
        f.write(str(no_loss[i]) + "\n")

