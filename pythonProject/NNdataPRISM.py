import os
import json
import jsonlines
import pickle
import sys

listOfTxtFiles = []
duplicates = []
# Finds the count of all images.
#parent_dir = '/Users/Vicente/PycharmProjects/Wireshark/pcaps'
#parent_dir = '/home/vicente/PycharmProjects/PRISMDATA/2021/Training/02/'
#parent_dir += str(sys.argv[1])
parent_dir = '/home/vicente/PycharmProjects/PRISMDATA/2021/Training/01/'

print(parent_dir)
#
for subdir, dirs, files in os.walk(parent_dir):
    for file in files:
        if file.endswith('PRISM.pickle'):
            temp = os.path.join(parent_dir, subdir)
            finalpath = (os.path.join(temp, file))
            listOfTxtFiles.append(finalpath)
Trace = 0
for file in listOfTxtFiles:
    jsonfile = file[:-7]
    jsonfile = jsonfile + ".jsonl"
    with open(jsonfile, 'w') as f:
        fpath = file
        masterPickle = []
        with open(file, 'rb') as fi:
            load = pickle.load(fi)
            Trace +=1
            #print(load)

            predict5 = []
            trainData20 = []
            if len(load[0]) > 0:
                finalload = load[0]
                print(len(load[0]))
                for i in range(int(len(finalload)-24)):
                    if len(finalload) > 25:
                        traindata = [x[1:] for x in finalload[i:i+20]]
                        predict = finalload[i+24]
                        preholder = []
                        windowPac = 0
                        #pre = predict[-1]

                        #windowPac += predict[-1]
                        #preholder.append(pre[-2])

                        #if sum(preholder) > 0:
                        #print(predict)
                        trainData20.append([traindata, predict])
                        #else:
                        #    trainData20.append([traindata, 0])
            for data in trainData20:
                # masterPickle.append([data,Trace])
                f.write(json.dumps([data, Trace]) + "\n")
            print(Trace)


