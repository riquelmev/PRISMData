import os
import sys
#Prints the count of all images.
parent_dir = '/home/vicente/PycharmProjects/PRISMDATA/'
parent_dir += str(sys.argv[1])
print(parent_dir)

count = 0
for subdir, dirs, files in os.walk(parent_dir):
    for fi in files:
        im = str.lower(fi)
        if im.endswith('.pcap.gz'):
            count+=1
print(count)

listOfFiles = []
listOfTxtFiles=[]
for subdir, dirs, files in os.walk(parent_dir):
    for fi in files:
        if fi.endswith('pcap.gz'):
            temp = os.path.join(parent_dir, subdir)
            finalpath = (os.path.join(temp, fi))
            listOfFiles.append(finalpath)

#print(listOfFiles)
for fi in listOfFiles:
    temp = fi[:-5]
    output = (temp + "_output.txt")
    var = ' tshark -r '+ fi + ' > ' + output + ' -Tfields -e "_ws.col.No." -e "_ws.col.Time" -e ip.src -e "_ws.col.Destination"  -e tcp.seq -e tcp.ack -e tcp.len -e tcp.port -Y tcp'
    print(var)
    os.system(var)

    #Number, Time, Source, Destintion, Seq, Ack, LenPort, TCP
