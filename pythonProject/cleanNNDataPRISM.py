import jsonlines
import numpy
import numpy as np
import json
import os
import math
import sys
#IN_FILE = '/home/vicente/storage/NNdataMlabspredict3.jsonl'
# for line in open(IN_FILE).readlines():
#     if line != '\n':
listOfTxtFiles = []
minRTTTooLow = 0
parent_dir = '/home/vicente/PycharmProjects/PRISMDATA/2021/Training/01'
#parent_dir += str(sys.argv[1])
print(parent_dir)
for subdir, dirs, files in os.walk(parent_dir):
    for fi in files:
        #fi = file.lower()
        #print(fi)
        if fi.endswith('PRISM.jsonl'):
            temp = os.path.join(parent_dir, subdir)
            print(temp)
            finalpath = (os.path.join(temp, fi))
            listOfTxtFiles.append(finalpath)
#jsonfile = '/home/vicente/storage/NNdata50ms_NewCleanHopefully.jsonl'
for file in listOfTxtFiles:
    jsonfile = file[:-6]
    jsonfile = jsonfile + "Clean.jsonl"
    minDataRTT = []
    windowholder= []

    with open(jsonfile, 'w') as r:
        with jsonlines.open(file) as f:
            for line in f.iter():
                if line != '\n':
                    # window = json.loads(line)
                    window = line
                    trace = window[1]
                    # print(trace)
                    window = window[0]
                    loss = window[1]
                    #print(loss)
                    window = window[0]

                    # for i in range(len(window)):
                    #     minDataRTT.append(window[i][0])

                    # if min(minDataRTT) < 0.002:
                    #     "MinRTT too low"
                    #     minRTTTooLow += 1
                    #     #continue?
                    #     break

                    for i in range(len(window)):
                        #print(type(window))
                        hasNan = True in (math.isnan(x) for x in window[i])
                        window[i] = [0 if math.isnan(x) else x for x in window[i]]
                        #window[i] = [x if not type(x) == str else 0 for x in window[i]]
                        if hasNan:
                            window[i].append(1)
                        else:
                            window[i].append(0)
                        #print(window)
                        # rttmin.append(window[i][0])
                        # rttmax.append(window[i][1])
                        # rttavg.append(window[i][2])
                    #print(loss)
                    window = [window, loss[-2]]
                    window = [window, trace]
                    #print(window)
                    windowholder.append(window)
            # if len(minDataRTT) == 0:
            #     continue
            for window in windowholder:
                r.write(json.dumps(window) + "\n")
print(len(listOfTxtFiles))