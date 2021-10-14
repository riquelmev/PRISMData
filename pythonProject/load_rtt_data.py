import json
import os
import jsonlines
import numpy as np
import collections
import pickle
IN_DIR = "RTTDATA"
#IN_FILE = "NNdataMlabs.jsonl"
#IN_FILE = "y_loss_data.jsonl"
IN_FILE = '/home/vicente/storage/NNdataMlabs2.jsonl'

with jsonlines.open(IN_FILE) as f:
    for line in f.iter():
        print(line[1])

#with open("/home/vicente/storage/data/y_loss_data.pickle", 'rb') as f:
#    load = pickle.load(f)
#print(collections.Counter(load))
