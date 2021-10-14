import os
import sys
import matplotlib.pyplot as plt
import csv
import pandas as pd
import statistics as stat
from scipy import stats
import pickle as pickle
import numpy as np
from collections import namedtuple

Packet_Tuple = namedtuple("Packet_Tuple", 'Num' 'Time' 'Source' 'Destination' 'Seq' 'Ack' 'LenPort' 'TCP')

listOfTxtFiles = []
#Finds the count of all images.
parent_dir = '/home/vicente/PycharmProjects/PRISMDATA/2021/Training/01/'
parent_dir += str(sys.argv[1])
print(parent_dir)
sourceFiles = {}
destFiles= {}
for subdir, dirs, files in os.walk(parent_dir):
    for fi in files:
        if fi.endswith('output.txt'):
            temp = os.path.join(parent_dir, subdir)
            print(temp)
            finalpath = (os.path.join(temp, fi))
            listOfTxtFiles.append(finalpath)
print(listOfTxtFiles)
print(len(listOfTxtFiles))
#loops through all files
count = 0
for file in listOfTxtFiles:
    print("Loop: " + file)
    count+=1
    print(count/len(listOfTxtFiles))
    with open(file) as dfile:
        data = dfile.read()
    tempMain = []
    sentances = data.splitlines()
    for sentance in sentances:
        sen = sentance.split()
        tempMain.append(sen)
    if len(tempMain) < 1:
        continue
    print(len(tempMain[0]))
    if len(tempMain[0]) != 8:
        continue

    #determines proper source, dest, and port
    current = tempMain[0]
    for i in range(len(tempMain)):
        if int(tempMain[i][4]) > int(current[4]):
            # print(tempMain[i])
            current = tempMain[i]
    if len(current) != 8:
        print('Not 8 elements')
        continue
    source = current[2]
    dest = current[3]
    port = current[7]

    #skims excess data
    sourcePackets = {}
    destPackets = {}
    Duplicates = []
    setOfElems = set()

    #Number, Time, Source, Destination, Seq, Ack, LenPort, TCP
    for packet in tempMain:
        #Packet_Tuple = namedtuple("Packet_Tuple", 'Num' 'Time' 'Source' 'Destination' 'Seq' 'Ack' 'LenPort' 'TCP')
        #Packet =
        if len(packet) == 8:
            packet_time = float(packet[1])
            source_of_packet = int(packet[2])
            length_of_packet = int(packet[6])
            seq = int(packet[4])
            Packet = Packet_Tuple()

            if packet[2] == source and int(packet[6]) > 0 and packet[7] == port:
                seq = int(packet[4])
                newPacket = [float(packet[1]), seq, int(packet[6])]
                #newPacket = [packet_time, seq, length_of_packet]
                if seq in sourcePackets:
                    sourcePackets[seq] = (newPacket, True)
                else:
                    sourcePackets[seq] = (newPacket, False)
            if packet[2] == dest and int(packet[6]) == 0:
                seq = int(packet[5])
                newPacket = [float(packet[1]), seq]
                #newPacket = [packet_time, seq]

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
            # spacket[0] = time, seq, len
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
        #print(RTTpacket)
        finalSourcePackets.append(RTTpacket)
    finalSourcePackets = finalSourcePackets[10:]
    finalPac = []

    if len(finalSourcePackets) > 500:
        start = finalSourcePackets[0][0]
        wasThereLoss = False


        window = []
        i = 0
        currentWindow = finalSourcePackets[0][0]
        windowSize = 1/60#0.05
        interval = 1/60 #0.01
        actualLossCount= set()
        bytesholder = []
        lostbytesholder = []
        rttholder = []
        while i < len(finalSourcePackets) and currentWindow < finalSourcePackets[-1][0]:
            while i < len(finalSourcePackets) and finalSourcePackets[i][0] < currentWindow + windowSize:
                window.append(finalSourcePackets[i])
                i += 1

            rtt = []
            timeDiv = []
            lossCount = 0
            bytes=0
            lostbytes = 0

            for tpacket in window:
                #[time, seq, len, rtt, Loss?]
                if tpacket[3] > 0:
                    rtt.append(tpacket[3])
                    timeDiv.append(tpacket[0])
                    bytes += tpacket[2]
                if tpacket[4] == True:
                    #print("found something")
                    lostbytes += tpacket[2]
                    if tpacket[0] not in actualLossCount:
                        lossCount += 1
                        actualLossCount.add(tpacket[0])

            if len(rtt) > 1:
                pac = [round(currentWindow, 2), min(rtt), max(rtt), stat.mean(rtt), stat.variance(rtt)]
            if len(rtt) == 1:
                pac = [round(currentWindow, 2), min(rtt), max(rtt), stat.mean(rtt), np.nan]
            if len(rtt) == 0:
                pac = [round(currentWindow, 2), np.nan, np.nan, np.nan, np.nan]
            if len(rtt) > 1:
                slope, intercept, r, p, se = stats.linregress(timeDiv, rtt)
                pac.append(slope)
                pac.append(intercept)
            if not len(rtt) > 1:
                pac.append(np.nan)
                pac.append(np.nan)
            pac.append(lossCount)
            pac.append(len(window))
            # print(packets)
            # print(pac)
            if lossCount > 0:
            #    print(pac)
                wasThereLoss = True
            finalPac.append(pac)
            bytesholder.append(bytes)
            lostbytesholder.append(lostbytes)
            if len(rtt) > 0:
                rttholder.append(stat.mean(rtt))
            else:
                rttholder.append(np.nan)

            currentWindow += interval
            while len(window) > 0 and window[0][0] <= currentWindow:
                window.pop(0)


        finalPac = [finalPac,wasThereLoss,len(finalSourcePackets), totalBytes,totalBytes2]
        #print(finalPac[0],finalPac[1])

        pickleFile = file[:-4]
        pickleFile = pickleFile + 'PRISM.pickle'
        #print(pickleFile)
        if len(rttholder) < 500:
            continue
        if np.nanmin(rttholder) < 0.002 or np.nanmax(rttholder) > 0.2:
            continue

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
        print(bytesholder)
        print(lostbytesholder)
        bytesfile = file[:-4]
        bytesfile = bytesfile + "Bytes0.16.txt"
        lostbytesfile = file[:-4]
        lostbytesfile = lostbytesfile + "LostBytes0.16.txt"
        rttfile = file[:-4]
        rttfile = rttfile + "AvgRTT0.16.txt"
        with open(bytesfile, 'w') as file:
            file.writelines("window size is " + str(windowSize))
            file.writelines("interval size is" + str(interval) + "\n")
            file.writelines(["%s\n" % item for item in bytesholder])
        with open(lostbytesfile, 'w') as file:
            file.writelines("window size is " + str(windowSize))
            file.writelines("interval size is" + str(interval) + "\n")
            file.writelines(["%s\n" % item for item in lostbytesholder])
        with open(rttfile, 'w') as file:
            file.writelines("window size is " + str(windowSize))
            file.writelines("interval size is" + str(interval) + "\n")
            file.writelines(["%s\n" % item for item in rttholder])


print(sourceFiles)
print(destFiles)
