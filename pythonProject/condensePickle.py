import pickle

import numpy as np

x_final =[]
y_final =[]
for i in range(0,20,5):
    print(i)
    xfile = '/home/vicente/storage/data/xtraining_data_SMOTE' + str(i) + '50ms.pickle'
    yfile = '/home/vicente/storage/data/ytraining_data_SMOTE' + str(i) + '50ms.pickle'
    with open(xfile, 'rb') as f:
        x_first = pickle.load(f)
    with open(yfile, 'rb') as f:
        y_first = pickle.load(f)
    print(type(x_first))
    print(type(y_first))


    xfile = '/home/vicente/storage/data/xtraining_data_SMOTE' + str(i+1) + '50ms.pickle'
    yfile = '/home/vicente/storage/data/ytraining_data_SMOTE' + str(i+1) + '50ms.pickle'
    with open(xfile, 'rb') as f:
        x_second = pickle.load(f)
    with open(yfile, 'rb') as f:
        y_second = pickle.load(f)
    print(type(x_second))
    print(type(y_second))

    x_1 = np.concatenate((x_first,x_second))
    y_1 = np.concatenate((y_first,y_second))

    xfile = '/home/vicente/storage/data/xtraining_data_SMOTE' + str(i+2) + '50ms.pickle'
    yfile = '/home/vicente/storage/data/ytraining_data_SMOTE' + str(i+2) + '50ms.pickle'
    with open(xfile, 'rb') as f:
        x_first = pickle.load(f)
    with open(yfile, 'rb') as f:
        y_first = pickle.load(f)

    xfile = '/home/vicente/storage/data/xtraining_data_SMOTE' + str(i+3) + '50ms.pickle'
    yfile = '/home/vicente/storage/data/ytraining_data_SMOTE' + str(i+3) + '50ms.pickle'
    with open(xfile, 'rb') as f:
        x_second = pickle.load(f)
    with open(yfile, 'rb') as f:
        y_second = pickle.load(f)

    x_2 = np.concatenate((x_first, x_second))
    y_2 = np.concatenate((y_first, y_second))

    xfile = '/home/vicente/storage/data/xtraining_data_SMOTE' + str(i+4) + '50ms.pickle'
    yfile = '/home/vicente/storage/data/ytraining_data_SMOTE' + str(i+4) + '50ms.pickle'
    with open(xfile, 'rb') as f:
        x_first = pickle.load(f)
    with open(yfile, 'rb') as f:
        y_first = pickle.load(f)

    x_3 = np.concatenate((x_2, x_first))
    y_3 = np.concatenate((y_2, y_first))
    x_2 = 0
    y_2 = 0
    x_final = np.concatenate((x_1, x_3))
    y_final = np.concatenate((y_1, y_3))
    x_1, x_3 = 0
    y_1, y_3 = 0
    print(x_final.shape)
    print(y_final.shape)
    print(str(int(i/5)))
    with open("/home/vicente/storage/data/x_condensed" + str(int(i/5))  +"50ms2.pickle", 'wb') as f:
        pickle.dump(x_final, f)
    with open("/home/vicente/storage/data/y_condensed" + str(int(i/5))  +"50ms2.pickle", 'wb') as f:
        pickle.dump(y_final, f)






