import jsonlines
import numpy
import numpy as np
import json
import os
#IN_FILE = '/home/vicente/storage/NNdataMlabspredict3.jsonl'
# for line in open(IN_FILE).readlines():
#     if line != '\n':
listOfTxtFiles = []
parent_dir = '/home/vicente/storage/31/2021/02/01'
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
    jsonfile = jsonfile + "CleanNoTag.jsonl"
    with open(jsonfile, 'w') as r:
        with jsonlines.open(file) as f:
            for line in f.iter():
                if line != '\n':
                    #window = json.loads(line)
                    window = line
                    trace = window[1]
                    #print(trace)
                    window = window[0]
                    loss = window[1]
                    window = window[0]
                    window = np.array(window)
                    #print(type(window))
                    # finalload = list(filter(lambda x: not (any([type(i) == str for i in x])), window))
                    # print(window)
                    # temp = np.array(packet[0])
                    # print(temp.dtype)
                    # print(temp)
                    #print(window)
                    if(window.dtype.type is np.str_):
                        #print(window)
                        #print(type(window))
                        for i in range(len(window)):
                            window[i] = np.where(window[i] == 'N/A', np.nan, window[i])
                        #print(window)
                        #print(type(window))
                        #continue
                        window=np.fromstring(window)
                    if np.isnan(window).any():
                        #print(window)
                        #window = numpy.resize(window, (20, 9))
                        #print(window)
                        #print(type(window))
                        #for i in range(len(window)):
                        #    if np.isnan(window[i]).any():
                                #print(window[i])
                        #        window[i][-1] = 1
                                #print(window[i])
                        #    else:
                         #       window[i][-1] = 0
                        window = numpy.nan_to_num(window)
                        window = window.tolist()
                        window = [window, loss]
                        window = [window, trace]
                    else:
                        #window = numpy.resize(window,(20,9))
                        #print(window)
                        #print(window.shape)
                        #for i in range(len(window)):
                            #print(window[i])
                        #    window[i][-1] = 0
                        window = window.tolist()
                        window = [window, loss]
                        window = [window, trace]
                    #print(window)
                    #window = list(window)
                    #print(window)
                    r.write(json.dumps(window) + "\n")
                    print(trace)
                    print(window)