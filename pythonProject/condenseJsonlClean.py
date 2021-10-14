import jsonlines
import numpy
import numpy as np
import json
import os
import sys

#IN_FILE = '/home/vicente/storage/NNdataMlabspredict3.jsonl'
# for line in open(IN_FILE).readlines():
#     if line != '\n':
listOfTxtFiles = []
parent_dir = '/home/vicente/PycharmProjects/PRISMDATA/2021/Training'
#parent_dir += str(sys.argv[1])
#print(parent_dir)
#model = keras.models.load_model('/home/vicente/storage/2classmodelTest0.82')

for subdir, dirs, files in os.walk(parent_dir):
    for fi in files:
        #fi = file.lower()
        #print(fi)
        if fi.endswith('Clean.jsonl'):
            temp = os.path.join(parent_dir, subdir)
            print(temp)
            finalpath = (os.path.join(temp, fi))
            listOfTxtFiles.append(finalpath)
with open('/home/vicente/storage/JsonlMaster.jsonl', 'w') as t:
    for file in listOfTxtFiles:
        with jsonlines.open(file) as f:
            for line in f.iter():
                if line != '\n':
                    t.write(json.dumps(line) + "\n")

