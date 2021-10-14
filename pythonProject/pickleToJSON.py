import os
import pickle
import jsonlines
import json

listOfTxtFiles = []
#Finds the count of all images.
parent_dir = '/home/vicente/storage/data'
for subdir, dirs, files in os.walk(parent_dir):
    for file in files:
        #fi = str.lower(file)
        for i in range(10):
            ending = 'SMOTE' + str(i) + '.pickle'
            #print(ending)
            if file.endswith(ending):
                temp = os.path.join(parent_dir, subdir)
                finalpath = (os.path.join(temp, file))
                listOfTxtFiles.append(finalpath)
#loops through all files
for file in listOfTxtFiles:
    print("Loop: " + file)
    jsonfile = file[:-6]
    jsonfile += 'jsonl'
    print(jsonfile)
    with open(file, 'rb') as f:
        load = pickle.load(f)
    with open(jsonfile, 'w') as f:
        for entry in load:
            #print(entry)
            f.write(json.dumps(entry.tolist()) + "\n")
    #with open(file) as dfile:
    #     data = dfile.read()