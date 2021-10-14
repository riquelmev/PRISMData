
#IN_FILE = '/home/vicente/storage/NNdata50ms_NewCleanHopefully.jsonl'
IN_FILE = '/home/vicente/storage/JsonlMasterTesting.jsonl'

strcount = 0

# fpath = os.path.join(IN_DIR, IN_FILE)
import jsonlines
import numpy as np
# for line in open(IN_FILE).readlines():
#     if line != '\n':
count = 0
losscount= 0
with jsonlines.open(IN_FILE) as f:
    for line in f.iter():
        if line != '\n':
            window = line
            trace = window[1]
            print(trace)
            window = window[0]
            loss = window[1]
            window = window[0]
            count +=1
            if count % 10000 == 0:
                print(count)
            if loss > 0:
                losscount +=1
            #window = np.array(window)
            # print(type(window))
            # finalload = list(filter(lambda x: not (any([type(i) == str for i in x])), window))
            # print(window)
            # temp = np.array(packet[0])
            # print(temp.dtype)
            # print(temp)
            # print(window)
            #if (window.dtype.type is np.str_):
            #    strcount +=1
#print(strcount)
print(losscount)
print(count)
print(losscount/count)