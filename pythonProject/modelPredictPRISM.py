import jsonlines
import numpy
import numpy as np
import json
import os
import keras
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import classification_report
#IN_FILE = '/home/vicente/storage/NNdataMlabspredict3.jsonl'
# for line in open(IN_FILE).readlines():
#     if line != '\n':
listOfTxtFiles = []
parent_dir = '/home/vicente/PycharmProjects/PRISMDATA/2021/Training/01/'
model = keras.models.load_model('/home/vicente/storage/2classmodelDUP0.82NewData')

for subdir, dirs, files in os.walk(parent_dir):
    for fi in files:
        #fi = file.lower()
        #print(fi)
        if fi.endswith('Clean.jsonl'):
            temp = os.path.join(parent_dir, subdir)
            print(temp)
            finalpath = (os.path.join(temp, fi))
            listOfTxtFiles.append(finalpath)

for file in listOfTxtFiles:
    x = []
    y = []
    with jsonlines.open(file) as f:
        for line in f.iter():
            if line != '\n':
                window = line
                trace = window[1]
                print(trace)
                window = window[0]
                loss = window[1]
                window = window[0]
                x.append(window)
                y.append(loss)
    print("len is:")
    print(len(x))
    print(len(y))
    newy = []
    for value in y:
        if value > 0:
            newy.append(1)
        else:
            newy.append(0)
    newy = tf.keras.utils.to_categorical(newy,2)
    #print(newy)
    #print(x)
    x = np.asarray(x)
    print(type(newy))
    print(type(x))
    print(newy.size)
    print(x.size)
    if (x.size == 0 or newy.size == 0):
        continue
    model.evaluate(x, newy)
    y_pred_ohe = model.predict(x,batch_size=1)  # shape=(n_samples, 12)
    # only necessary if output has one-hot-encoding, shape=(n_samples)
    # y_pred_labels = np.argmax(y_pred_ohe, axis=1)
    y_pred = np.argmax(y_pred_ohe, axis=1)
    y_cat = np.argmax(newy, axis=1)
    textfile = file[:-16]
    actualfile = textfile + "actual.txt"
    textfile = textfile + "prediction.txt"

    with open(textfile,'w') as writefile:
        writefile.writelines(["%s\n" % item for item in y_pred_ohe])
    with open(actualfile,'w') as writefile:
        writefile.writelines(["%s\n" % item for item in y_cat])




#     # print(y_pred_labels)
#     # print(y_cat)
#     print(y_pred_ohe)
#     print(y_pred_ohe)
#     print(y_cat)
#     loss = []
#     no_loss = []
#     values = []
#     y_pred_labels = []
#     for value in y_pred_ohe:
#         if float(value[1]) == 0 or float(value[0]) / float(value[1]) > 5:
#             y_pred_labels.append(0)
#         else:
#             y_pred_labels.append(1)
#     y_pred_labels = np.asarray(y_pred_labels)
#     matrix = metrics.confusion_matrix(y_cat, y_pred_labels)
#     # matrix = metrics.confusion_matrix(y_val.argmax(axis=1), pred.argmax(axis=1))
#
#     print(matrix)
#     print(matrix[1][1] / (matrix[1][1] + matrix[1][0]))
#     print(matrix[0][0] / (matrix[0][0] + matrix[0][1]))
#     loss_recall = matrix[1][1] / (matrix[1][1] + matrix[1][0])
#     noloss_recall = matrix[0][0] / (matrix[0][0] + matrix[0][1])
#     loss.append(loss_recall)
#     no_loss.append(noloss_recall)
#
# print(classification_report(y_cat, y_pred_labels))




