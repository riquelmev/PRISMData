import jsonlines
import numpy
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
IN_FILE = '/home/vicente/storage/NNdataMlabspredict3.jsonl'
# for line in open(IN_FILE).readlines():
#     if line != '\n':
jsonfile = '/home/vicente/storage/NNdata50ms_NewCleanHopefully.jsonl'
packetsLoss = 0
windowsLoss=0
count = 0
with open(jsonfile, 'w') as r:
    with jsonlines.open(IN_FILE) as f:
        counter = 0
        # rttmin=[]
        # rttmax=[]
        # rttavg=[]
        for line in f.iter():
            counter +=1
            # if counter > 10000:
            #     break
            if line != '\n':
                #window = json.loads(line)
                window = line
                trace = window[1]
                #print(trace)
                window = window[0]
                loss = window[1]
                window = window[0]
                #window = np.array(window)
                #print(type(window))
                # finalload = list(filter(lambda x: not (any([type(i) == str for i in x])), window))
                # print(window)
                # temp = np.array(packet[0])
                # print(temp.dtype)
                # print(temp)
                #print(window)
                minDataRTT = []
                for i in range(len(window)):

                    hasNan = True in (type(x) == str for x in window[i])
                    window[i] = [x if not type(x) == str else 0 for x in window[i]]
                    if hasNan:
                        window[i].append(1)
                    else:
                        window[i].append(0)
                    minDataRTT.append(window[i][0])
                    # rttmin.append(window[i][0])
                    # rttmax.append(window[i][1])
                    # rttavg.append(window[i][2])


                window = [window, loss]
                window = [window, trace]








                # if (window.dtype.type is np.str_):
                #     # print(window)
                #     # print(type(window))
                #     for i in range(len(window)):
                #         window[i] = np.where(window[i] == 'N/A', np.nan, window[i])
                #     #print(window)
                #     # print(type(window))
                #     # continue
                #     window = np.frombuffer(window, dtype=float)
                #     #print(window)
                #     #window = np.fromstring(window)
                # if np.isnan(window).any():
                #     #print(window)
                #     window = numpy.resize(window, (20, 9))
                #     # print(window)
                #     # print(type(window))
                #     windowsLoss += 1
                #     for i in range(len(window)):
                #         if np.isnan(window[i]).any():
                #             packetsLoss += 1
                #             # print(window[i])
                #             window[i][-1] = 1
                #             print(window[i])
                #             # print(window[i])
                #         else:
                #             window[i][-1] = 0
                #     window = numpy.nan_to_num(window)
                    #window = window.tolist()
                # window = [window, loss]
                # window = [window, trace]
                # else:
                #     window = numpy.resize(window, (20, 9))
                #     # print(window)
                #     # print(window.shape)
                #     for i in range(len(window)):
                #         # print(window[i])
                #         window[i][-1] = 0
                #     window = window.tolist()
                #     window = [window, loss]
                #     window = [window, trace]
                #print(window)
                #window = list(window)
                #print(window)
                if min(minDataRTT) < 0.002:
                    continue
                count +=1
                r.write(json.dumps(window) + "\n")
                #print(trace)
print("packets",packetsLoss)
print("windows",windowsLoss)
print(count)
#sns.histplot(x=rttmin)
# rttmin = [x for x in rttmin if x < .02 and x != 0 and x > 0.005]
# rttmax = [x for x in rttmax if x < .02 and x != 0 and x > 0.005]
# rttavg = [x for x in rttavg if x < .02 and x != 0 and x > 0.005]
# plt.hist(rttmin, label="min rtt")
# plt.figure()
# plt.hist(rttmax, label="max rtt")
# plt.figure()
# plt.hist(rttmin, label="avg rtt")
#
# plt.show()

