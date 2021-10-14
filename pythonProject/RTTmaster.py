import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
import statistics as stat
from scipy import stats
import pickle as pickle

listOfTxtFiles = []
#Finds the count of all images.
#parent_dir = '/Users/Vicente/PycharmProjects/Wireshark/pcaps'
parent_dir = '/home/vicente/PycharmProjects/PRISMDATA'

sourceFiles = {}
destFiles= {}
for subdir, dirs, files in os.walk(parent_dir):
    for fi in files:
        #fi = file.lower()
        #print(fi)
        if fi.endswith('.txt'):
            temp = os.path.join(parent_dir, subdir)
            print(temp)
            finalpath = (os.path.join(temp, fi))
            listOfTxtFiles.append(finalpath)
print(listOfTxtFiles)
#loops through all files
for file in listOfTxtFiles:
    print("Loop: " + file)
    with open(file) as dfile:
        data = dfile.read()
    tempMain = []
    sentances = data.splitlines()
    for sentance in sentances:
        sen = sentance.split()
        tempMain.append(sen)
    print(len(tempMain[0]))
    if len(tempMain[0]) != 8:
        continue

    #determines proper source, dest, and port
    current = tempMain[0]
    for i in range(len(tempMain)):
        if int(tempMain[i][4]) > int(current[4]):
            # print(tempMain[i])
            current = tempMain[i]
    source = current[2]
    dest = current[3]
    port = current[7]

    #skims excess data
    sourcePackets = {}
    destPackets = {}
    Duplicates = []
    setOfElems = set()
    for packet in tempMain:
        if packet[2] == source and int(packet[6]) > 0 and packet[7] == port:
            seq = int(packet[4])
            newPacket = [float(packet[1]), seq, int(packet[6])]
            if seq in sourcePackets:
                sourcePackets[seq] = (newPacket, True)
            else:
                sourcePackets[seq] = (newPacket, False)
        if packet[2] == dest and int(packet[6]) == 0:
            seq = int(packet[5])
            newPacket = [float(packet[1]), seq]
            if seq not in destPackets:
                destPackets[seq] = newPacket
    #print(sourcePackets)
    finalSourcePackets = []
    totalBytes = 0
    totalBytes2 = 0
    for seq in sorted(sourcePackets.keys()):
        spacket = sourcePackets[seq]
        RTTpacket = list(spacket[0])
        # print(spacket)
        if spacket[1] == False:
            ack = seq + int(spacket[0][2])
            totalBytes += int(spacket[0][2])
            if ack in destPackets:
                rtt = (float(destPackets[ack][0]) - float(spacket[0][0]))
                RTTpacket.append(rtt)

                # if rtt > 0.06:
                # print(seq)
            else:
                RTTpacket.append(0)
            RTTpacket.append(False)
        else:
            RTTpacket.append(0)
            RTTpacket.append(True)
        totalBytes2 += int(spacket[0][2])
        finalSourcePackets.append(RTTpacket)
    finalSourcePackets = finalSourcePackets[10:]
    #print(finalSourcePackets)
    #for packet in finalSourcePackets:
    #    if packet[4] == True:
    #        print(packet)

    # finished = False
    # window = 0.05
    # start = 0.00
    # end = start + window
    # increment = 0.01
    # temp=[]
    finalPac = []
    # final = []
    if len(finalSourcePackets) > 0:
        start = finalSourcePackets[0][0]
        # end = finalSourcePackets[-1][0]
        # lossCount = 0
        wasThereLoss = False


        window = []
        i = 0
        currentWindow = finalSourcePackets[0][0]
        windowSize = 0.05
        interval = 0.01
        actualLossCount= set()
        while i < len(finalSourcePackets) and currentWindow < finalSourcePackets[-1][0]:
            while i < len(finalSourcePackets) and finalSourcePackets[i][0] < currentWindow + windowSize:
                window.append(finalSourcePackets[i])
                i += 1

            rtt = []
            timeDiv = []
            lossCount = 0
            for tpacket in window:
                #print(tpacket[4])
                if tpacket[3] > 0:
                    rtt.append(tpacket[3])
                    timeDiv.append(tpacket[0])
                if tpacket[4] == True:
                    #print("found something")
                    if tpacket[0] not in actualLossCount:
                        lossCount += 1
                        actualLossCount.add(tpacket[0])

            if len(rtt) > 1:
                pac = [round(currentWindow, 2), min(rtt), max(rtt), stat.mean(rtt), stat.variance(rtt)]
            if len(rtt) == 1:
                pac = [round(currentWindow, 2), min(rtt), max(rtt), stat.mean(rtt), 'N/A']
            if len(rtt) == 0:
                pac = [round(currentWindow, 2), 'N/A', 'N/A', 'N/A', 'N/A']
            if len(rtt) > 1:
                slope, intercept, r, p, se = stats.linregress(timeDiv, rtt)
                pac.append(slope)
                pac.append(intercept)
            if not len(rtt) > 1:
                pac.append('N/A')
                pac.append('N/A')
            pac.append(lossCount)
            pac.append(len(window))
            # print(packets)
            # print(pac)
            if lossCount > 0:
            #    print(pac)
                wasThereLoss = True
            finalPac.append(pac)

            currentWindow += interval
            while len(window) > 0 and window[0][0] <= currentWindow:
                window.pop(0)


        finalPac = [finalPac,wasThereLoss,len(finalSourcePackets), totalBytes,totalBytes2]
        print(finalPac[0],finalPac[1])

        pickleFile = file[:-4]
        pickleFile = pickleFile + 'PRISM.pickle'
        print(pickleFile)
        with open(pickleFile, 'wb') as f:
            pickle.dump(finalPac, f)

        if source in sourceFiles.keys():
            sourceFiles[source] = sourceFiles[source] + 1
        else:
            sourceFiles[source] = 1
        dest = dest[:3]
        if dest in destFiles:
            destFiles[dest] = destFiles[dest] + 1
        else:
            destFiles[dest] = 1

print(sourceFiles)
print(destFiles)