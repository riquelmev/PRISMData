


import tensorflow as tf

import pickle as pickle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
import pylab as pl
from tqdm import tqdm
import json
#file = '/SNAPPcapAnalysis/NNdataMlabs.pickle'
# file = 'NNdataMlabs.pickle'
# with open(file, 'rb') as f:
#     load = pickle.load(f)
# 
# print("loaded")
#file = '/home/vicente/PycharmProjects/pythonProject/RTTDATA/RTTOUT_1626295365.jsonl'





x_raw = []
y_raw= []

#print(load[0][0])
#print(load[0][1])
bins = []
for i in range(3):
    bins.append([])
count = 0
total_loss = 0
total = 0

x_loss = []
y_loss =[]

x_no_loss = []
y_no_loss = []
#IN_FILE = "NNdataMlabs.jsonl"
#IN_FILE = '/home/vicente/storage/NNdata50ms_NewCleanHopefully.jsonl'
IN_FILE = '/home/vicente/storage/JsonlMaster.jsonl'


# fpath = os.path.join(IN_DIR, IN_FILE)
import jsonlines
# for line in open(IN_FILE).readlines():
#     if line != '\n':
with jsonlines.open(IN_FILE) as f:
    for line in f.iter():
        if line != '\n':
            #window = json.loads(line)
            window = line

            #print(parsed_line)

    #for window in tqdm(load):
            trace = window[1]
            if count % 10000 == 0:
                print(count)
            window = window[0]
            count +=1
            #print(window)
            #finalload = list(filter(lambda x: not (any([type(i) == str for i in x])), window))
            #print(window)
            #temp = np.array(packet[0])
            #print(temp.dtype)
            #print(temp)
            if np.isnan(np.sum(np.array(window[0]))) or np.isnan(np.sum(np.array(window[0]))):
                print("removed something", trace)
                continue
            #if count > 6000000:
            #    break
            #x_raw.append(window[0])
            lossRate = 0.01
            if window[1] > 0:
                x_loss.append(window[0])
                total_loss = total_loss + 1
                total = total + 1
                #count = count + 1
                #y_raw.append(1)
                holder = 3
                #while window[1] < lossRate * holder:
                #    holder = holder - 1
                #holder = holder - 1
                lossInterval = round(window[1],3)
                lossInterval = int(lossInterval * 100)
                y_loss.append(1)
                # if lossInterval < 1:
                #     y_loss.append(1)
                # elif lossInterval < 3:
                #     y_loss.append(2)
                # else:
                #     y_loss.append(3)
            else:
                total = total +1
            #window = None
    #y_raw.append(window[1])


# for i in range(3):
#     #print(bins[i])
#     print("LossRate:",i * 0.05 , "-", (i +1) *0.05, sum(bins[i]), (sum(bins[i])/count))
# print("Percent of windows with loss",total_loss / total)
# print(total_loss)
# print(total)

x_loss = np.asarray(x_loss)
y_loss = np.asarray(y_loss)
num_classes = 4
#num_classes = len(set(y_raw))
#y_loss = tf.keras.utils.to_categorical(y_loss, num_classes)
print(num_classes)
print(y_loss)
print(y_loss.shape)

with open("/home/vicente/storage/data/x_loss_data50msListFilter.pickle", 'wb') as f:
    pickle.dump(x_loss, f)

with open("/home/vicente/storage/data/y_loss_data50msListFilter.pickle", 'wb') as f:
    pickle.dump(y_loss, f)
print("starting pickle")
#
# with open("x_loss_data.jsonl", 'w') as f:
#     for entry in x_loss:
#         json.dump(entry.tolist(), f)
#         f.write('\n')
#
# print("starting y data")
# with open("y_loss_data.jsonl", 'w') as f:
#     for entry in y_loss:
#         json.dump(entry.tolist(), f)
#         f.write('\n')

x_loss = None
y_loss = None
#
# x_no_loss = np.asarray(x_no_loss)
# y_no_loss = np.asarray(y_no_loss)
# #num_classes = len(set(y_raw))
# y_no_loss = tf.keras.utils.to_categorical(y_no_loss, num_classes)
# print(num_classes)
# print(y_no_loss)
#
# x_no_loss = np.array_split(x_no_loss,10)
# y_no_loss = np.array_split(y_no_loss,10)
# y_no_loss = y_no_loss[0]
#
# with open("y_no_loss_data.jsonl", 'w') as f:
#     for entry in y_no_loss:
#         json.dumps(entry, f)
#         f.write('\n')
# y_no_loss = None
#
# print("done with y no loss")
#
# for i in range(10):
#     file = i
#     with open("x_no_loss_data" + str(i) + ".jsonl", 'w') as f:
#         print("Starting loop:",i)
#         for entry in x_no_loss[i]:
#             json.dumps(entry, f)
#             f.write('\n')
#
# y_raw = np.asarray(y_raw)
# num_classes = len(set(y_raw))
# ytry = tf.keras.utils.to_categorical(y_raw, num_classes)
# #ytry = ytry.tolist()
#
# x_raw = np.asarray(x_raw)
# # print(len(ytry))
# # print(len(x_raw))
# # print(type(ytry))
# # print(type(x_raw))
#
#
#
# y_raw = None
# load = None



#print(x_raw[:10])
#print(y_raw[:10])
#print(len(x_raw[0][0]))

# x_train, x_val, y_train, y_val = train_test_split(x_raw, ytry, test_size=0.2, stratify=y_raw)
#
# with open("xtraining_data.pickle", 'wb') as f:
#     pickle.dump(x_train, f)
#
# with open("ytraining_data.pickle", 'wb') as f:
#     pickle.dump(y_train, f)
#
# with open("xtesting_data.pickle", 'wb') as f:
#     pickle.dump(x_val, f)
#
# with open("ytesting_data.pickle", 'wb') as f:
#     pickle.dump(y_val, f)
#
# print("DONE PICKLING")
# x_raw = None
# ytry = None
# print(x_train.shape)
# print(y_train.shape)
print(count)