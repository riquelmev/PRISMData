"""
This script finds all the TXT files in AllTraces, and loading them into a
JSONL (JSON lines) file
"""
# Builtin imports
import json
import os
from typing import List
from multiprocessing import Process, Pool, Queue
import time
import numpy as np

# External imports
import pandas as pd
import statistics as stat
from scipy import stats
from tqdm import tqdm

_TRACE_FOLDER = "/home/vicente/storage/"
_OUT_DIR = "RTTDATA"
_OUT_FILE = "RTT_{0}.jsonl"

def find_files() -> List:
    """
    Find all txt files in _TRACE_FOLDER
    """
    all_txt_files = []
    for subdir, dirs, files in os.walk(_TRACE_FOLDER):
        all_txt_files += [os.path.join(_TRACE_FOLDER, subdir, f) for f in files if f.endswith('.txt')]
    return all_txt_files

def write_task(file_queue: Queue, fname: str, process_num: int):
    """
    A process that will run continuously, writing data out
    :return:
    """
    i = 0
    with open(fname, 'w') as o:
        while True:
            next_obj = file_queue.get()
            if type(next_obj) == str and next_obj == "DONE":
                break
            else:
                assert type(next_obj) == list
                o.write(f"{json.dumps(next_obj)}\n")
                # Every 1000 lines, flush the data
                if i % 1000 == 0:
                    o.flush()
                if i % 100 == 0:
                    print(f"Write Process {process_num} - wrote {i} lines")
            i += 1
            
            
def ingest_data(file_name):
    fdata = open(file_name).read()
    lines = []
    # "_ws.col.No." -e "_ws.col.Time" -e ip.src -e "_ws.col.Destination"  -e tcp.seq -e tcp.ack -e tcp.len -e tcp.port -Y tcp'
    #  Col 0 no #         Col 1 Time     Col 2 src     Col 3 DST            Col 4 SEQ   COl 5 ACK  COL 6 LEN  COL 7 port
    df = pd.DataFrame([l.split() for l in fdata.splitlines()]).astype({4: "int32"}).astype({6: "int32"})
    max_seq = df[4].idxmax()
    r = df.iloc[max_seq]
    source = r[2]
    dest = r[3]
    port = r[7]
    source_df = df.loc[(df[2] == source) & (df[6] > 0) & (df[7] == port)]
    d = dir(source_df)
    dest_df = df.loc[(df[2] == dest) & (df[6] == 0)]
    # After filtering out unneeded data, iterate through the remaining, pulling into source and dest
    source_arr = source_df.to_numpy()
    dest_arr = dest_df.to_numpy()
    source_packets = {}
    dest_packets = {}
    for pack in source_arr:
        seq = pack[4]
        new_packet = [pack[1], seq, pack[6]]
        if seq in source_packets:
            source_packets[seq] = (new_packet, True)
        else:
            source_packets[seq] = (new_packet, False)

    for pack in dest_arr:
        ack = int(pack[5])
        new_packet = [pack[1], ack]
        if ack not in dest_packets:
            dest_packets[ack] = new_packet
            
    return source_packets, dest_packets


def compute_rtt(source_packets, dest_packets):
    final_source_packets = []
    total_bytes = 0
    oth_total_bytes = 0
    for seq in sorted(source_packets.keys()):
        spacket = source_packets[seq]
        RTTpacket = list(spacket[0])
        RTTpacket[0] = float(RTTpacket[0])
        # print(spacket)
        if spacket[1] == False:
            ack = seq + int(spacket[0][2])
            total_bytes += int(spacket[0][2])
            if ack in dest_packets:
                rtt = (float(dest_packets[ack][0]) - float(spacket[0][0]))
                RTTpacket.append(rtt)

            else:
                RTTpacket.append(0)
            RTTpacket.append(False)
        else:
            RTTpacket.append(0)
            RTTpacket.append(True)
        oth_total_bytes += int(spacket[0][2])
        final_source_packets.append(RTTpacket)
    final_source_packets = final_source_packets[10:]
    return final_source_packets, total_bytes, oth_total_bytes

def compute_window_loss(rtt_source_packets):
    wasThereLoss = False
    finalPac = []
    # final = []
    first_l = len(rtt_source_packets)
    if len(rtt_source_packets) > 0:
        start = rtt_source_packets[0][0]
        window = []
        i = 0
        currentWindow = rtt_source_packets[0][0]
        windowSize = 0.05
        interval = 0.01
        actualLossCount = set()
        while i < len(rtt_source_packets) and currentWindow < rtt_source_packets[-1][0]:
            while i < len(rtt_source_packets) and rtt_source_packets[i][0] < currentWindow + windowSize:
                window.append(rtt_source_packets[i])
                i += 1

            rtt = []
            timeDiv = []
            lossCount = 0
            for tpacket in window:
                # print(tpacket[4])
                if tpacket[3] > 0:
                    rtt.append(tpacket[3])
                    timeDiv.append(tpacket[0])
                if tpacket[4] == True:
                    # print("found something")
                    if tpacket[0] not in actualLossCount:
                        lossCount += 1
                        actualLossCount.add(tpacket[0])

            pac = []

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

            currentWindow += interval
            while len(window) > 0 and window[0][0] <= currentWindow:
                window.pop(0)
    l = len(rtt_source_packets)
    return finalPac, wasThereLoss


def process_file(file_name: str):
    # Ingest and filter the data
    try:
        t_start = time.time()
        source_packets, dest_packets = ingest_data(file_name)
        rtt_source_packets, main_byte_count, oth_byte_count = compute_rtt(source_packets, dest_packets)
        final_pac, was_there_loss = compute_window_loss(rtt_source_packets)
        write_pac = [final_pac, was_there_loss, len(rtt_source_packets), main_byte_count, oth_byte_count]
        process_file.q.put(write_pac)
        time_end = time.time()
        return True, time_end-t_start
    except (KeyError, ValueError):
        return False, "N/A"

def task_init(file_queue):
    process_file.q = file_queue


def coalesce_files(filenames):
    new_filename = os.path.join(_OUT_DIR, f"RTTOUT_{int(time.time())}.jsonl")
    with open(new_filename, 'w') as o_write:
        for f in filenames:
            with open(f) as op:
                for li in tqdm(op.readlines()):
                    o_write.write(f"{li}\n")

def main():
    print("Starting & finding files...")
    files = find_files()
    file_queue = Queue()
    print("Starting write process...")
    write_num = max(int(os.cpu_count() / 2), 1)
    # Start the process that will continuously write to the file queue
    written_files = []
    write_processes = []
    print(f"Starting {write_num} write processes...")
    for i in range(write_num):
        out_fname = os.path.join(_OUT_DIR, _OUT_FILE.format(i))
        write_process = Process(target=write_task, args=(file_queue,
                                                         out_fname,
                                                         i, ))
        written_files.append(out_fname)
        write_process.start()
        write_processes.append(write_process)
    cpu_count = os.cpu_count() * 2
    # The average processing time of a sample of these files
    prev_avg_time = 1.7427909488373574
    est_time = (prev_avg_time * len(files)) / cpu_count
    total_minutes = est_time / 60
    seconds = round(est_time % 60, 2)
    hours = round(total_minutes / 60, 2)
    minutes = round(total_minutes % 60, 2)

    print(f"Starting file workers with {cpu_count} workers, processing {len(files)} files, est time is {hours} hours, "
          f"{minutes} minutes, "
          f"{seconds} seconds...")
    p = Pool(cpu_count, initializer=task_init, initargs=(file_queue, ))
    res = p.map(process_file, files)
    p.close()
    p.join()
    processed = len([i for i in res if i[0]])
    skipped = len([s for s in res if not s[0]])
    #times = list(filter(lambda q: type(q) != str, [s[1] for s in res]))
    #avg_time = len(times) / sum(times)
    print(f"Done processing, processed {processed}, skipped {skipped}, waiting for IO...")
    for _ in range(write_num):
        file_queue.put("DONE")
    for write_process in write_processes:
        write_process.join()
    print("Coalescing final data...")
    coalesce_files(written_files)
    print("Done!")


if __name__ == "__main__":
    main()