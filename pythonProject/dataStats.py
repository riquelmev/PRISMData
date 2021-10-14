import pickle

windows_with_loss = 0
input_with_loss = 0
file = '/home/vicente/storage/data/x_validation50ms.pickle'
#file = "/home/vicente/storage/data/x_condensed050ms2.pickle"
#file = "/home/vicente/storage/data/x_loss_data50ms.pickle"
#with open("/home/vicente/storage/data/x_condensed" + str(int(i / 5)) + "50ms2.pickle", 'wb') as f:

with open(file, 'rb') as f:
    x_val = pickle.load(f)

for window in x_val:
    badData = False
    for packet in window:
        #print(packet)
        if packet[-1] == 0:
            print(packet)
            windows_with_loss += 1
            badData = True
            print(window)
    if badData:
        input_with_loss +=1

print(x_val.shape)
print("individual windows with loss:",windows_with_loss)
print("input with loss", input_with_loss)
