


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
IN_FILE = '/home/vicente/storage/NNdataMlabs2.jsonl'

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
            print(trace)
            window = window[0]
            #print(window)
            #temp = np.array(packet[0])
            #print(temp.dtype)
            #print(temp)
            if np.isnan(np.sum(np.array(window[0]))) or np.isnan(np.sum(np.array(window[1]))):
                continue
            if trace > 200000:
                break
            #x_raw.append(window[0])
            lossRate = 0.0
            if window[1] == 0 and trace < 100000:
                #y_raw.append(0)
                x_no_loss.append(window[0])
                y_no_loss.append(0)
                total = total +1

            elif window[1] > 0 and trace < 100000:
                x_no_loss.append(window[0])
                y_no_loss.append(1)

    #y_raw.append(window[1])
print("starting x asrray")
x_no_loss = np.asarray(x_no_loss)
y_no_loss = np.asarray(y_no_loss)

num_classes = 4
#y_loss = tf.keras.utils.to_categorical(y_loss, num_classes)

print("starting pickle")
#x_no_loss = np.array_split(x_no_loss,10)
#size= []
print("Iteration: X")
with open("/home/vicente/storage/data/x_no_loss_2catNoUPSMOTE.pickle", 'wb') as f:
    pickle.dump(x_no_loss, f)
#size.append(x_no_loss.shape[0])
x_no_loss = 0

print("Starting y asarray")
print("Iteration: Y")
with open("/home/vicente/storage/data/y_no_loss_2catNoUPSMOTE.pickle", 'wb') as f:
    pickle.dump(y_no_loss, f)
#size.append(x_no_loss.shape[0])
y_no_loss = 0