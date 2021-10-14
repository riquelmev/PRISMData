import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
import statistics as stat
from scipy import stats
import pickle as pickle
import numpy as np
#import seaborn as sns
import scipy.stats as stats
import json
import jsonlines



listOfTxtFiles = []
duplicates = []

IN_DIR = "RTTDATA"
#IN_FILE = "RTTOUT_1626295365.jsonl"
IN_FILE = "RTTOUT_1627400820.jsonl"

jsonfile = '/home/vicente/storage/NNdataMlabspredict3.jsonl'
with open(jsonfile, 'w') as f:


    fpath = os.path.join(IN_DIR, IN_FILE)
    Trace = 0
    masterPickle = []
    for line in open(fpath).readlines():
        if line != '\n':
            load = json.loads(line)
            Trace +=1


    # for file in listOfTxtFiles:
    #     print("Loop: " + file)
    #     with open(file, 'rb') as f:
    #         load = pickle.load(f)

        predict5 = []
        trainData20 = []
        if len(load[0]) > 0:
            #finalload = list(filter(lambda x: not (any([type(i) == str for i in x])), load[0]))
            finalload = load[0]
            #finalload = [[i for i in l if type(i) != str] for l in load[0] if any([type(i) != str for i in l])]
            print(len(load[0]))
            for i in range(int(len(finalload)/20)):
                if len(finalload) > 25:
                    traindata = finalload[:20]
                    #if finalload[]
                    finaltrain= []
                    for train in traindata:
                        train1 = train[1:]
                        #train1 = train1[:-2]
                        #print(train1)
                        finaltrain.append(train1)

                    #print(traindata)
                    finalload = finalload[20:]
                    predict = finalload[:5]
                    #print("This is finaltrain",finaltrain)
                    #print("This is predict",predict)
                    preholder = []
                    windowPac = 0
                    pre = predict[-1]
                    #for pre in predict:
                    #Need to have loss and paccket count
                    windowPac += pre[-1]
                    preholder.append(pre[-2])
                    #print(preholder)
                    #train1 = traindata[:-2]
                    if sum(preholder) > 0:
                        trainData20.append([finaltrain,sum(preholder)/windowPac])
                    else:
                        trainData20.append([finaltrain, 0])
        for data in trainData20:
            #masterPickle.append([data,Trace])
            f.write(json.dumps([data, Trace]) + "\n")
        print(Trace)
        # for window in masterPickle:     # print(window[1])

    # loop = 9
    # for packet in masterPickle:
    #     if packet[1] > 0:
    #         for i in range(loop):
    #             duplicates.append(packet)
    #     #if packet[1] > 0:
    # print(len(duplicates))
    #
    # print(masterPickle[0])
    # print(duplicates[0])
    # print(len(masterPickle))
    # for packet in duplicates:
    #     masterPickle.append(packet)
    # #for packet in masterPickle:
    # #    print(packet[1])
    #
    # print(len(masterPickle))
    print("beginning to pickle")
    #pickleFile = 'NNdataMlabs.pickle'
    #jsonfile = '/home/vicente/storage/NNdataMlabs2.jsonl'
    #print(pickleFile)
    # with open(pickleFile, 'wb') as f:
    #     pickle.dump(masterPickle, f)

    # with open(jsonfile, 'w') as f:
    #     for entry in masterPickle:
    #         print(entry[1])
    #         json.dump(entry, f)
    #         f.write('\n')

    # with jsonlines.open(jsonfile, 'w') as writer:
    #     writer.write_all(masterPickle)





