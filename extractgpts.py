import argparse
import multiprocessing
import os
import socket
from scapy.all import rdpcap


import dpkt
import numpy as np
from tqdm import tqdm
import json
import imgvalid.validator as imgvalidator

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_name", type=str, default="gpts", help="dataset name")
args = parser.parse_args()

dataset_name = args.dataset_name


# get path list
def get_path_list(dataset_name: str):
    print("Begin get path list")
    captured_dir = os.path.join("data", "captured", dataset_name)
    paths = os.walk(captured_dir, followlinks=True)

    path_list = []
    for path, dir_lst, file_lst in paths:
        # should be a url dir
        if not "instance" in os.path.basename(path):
            continue
        # should have pcap file
        pcap_file = os.path.join(path, "tcp.pcap")
        if not os.path.exists(pcap_file):
            continue
        # should have result.json
        result_json = os.path.join(path, "result.json")
        if not os.path.exists(result_json):
            continue
        result = json.load(open(result_json))
        if not( "msg" in  result and "assistant" in result["msg"] and len(''.join(result["msg"]["assistant"]))>100):
            continue
        path_list.append(
            {
                "path": path,
                "pcap_file": pcap_file,
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "label": result["name"],
            }
        )
    print("Finish get path list")
    return path_list


path_list = get_path_list(dataset_name)
print("path_list length: {}".format(len(path_list)))
task_list = path_list


def extract_pcap(pcap_file: str):

    web_ip_sequence_dict = {}
    
    ips = set()
    # detect host ip
    host_ip = None
    for packet in rdpcap(pcap_file):
        if packet.haslayer("TCP"):
            ip = packet.getlayer("IP")
            if len(ips)>2:
                if ip.src in ips and ip.dst not in ips:
                    host_ip = ip.src
                    break
                elif ip.dst in ips and ip.src not in ips:
                    host_ip = ip.dst
                    break
            ips.add(ip.src)
            ips.add(ip.dst)
    assert host_ip is not None, "host ip not found"

    for packet in rdpcap(pcap_file):
        if packet.haslayer("TCP"):
            ip = packet.getlayer("IP")
            tcp = packet.getlayer("TCP")
            record_length = len(packet.getlayer("TCP").payload)
            if record_length == 0:
                continue
            if (
                record_length == 6
                and packet.getlayer("TCP").payload.load == b"\x00\x00\x00\x00\x00\x00"
            ):
                continue
            ip_set = {ip.src, ip.dst}
            assert host_ip in ip_set, "ip.src: {}, ip.dst: {}, host_ip: {}".format(
                ip.src, ip.dst, host_ip
            )
            # (web_ip,) = ip_set - {host_ip} # 
            web_ip = "0.0.0.0"
            if not web_ip in web_ip_sequence_dict:
                web_ip_sequence_dict[web_ip] = list()
            ts = float(packet.time)
            web_ip_sequence_dict[web_ip].append((ts, ip.src, ip.dst, tcp.seq, record_length))

    return web_ip_sequence_dict, host_ip


def extract_pcap_client(pcap_file: str):
    CLIENT_TO_SERVER, SERVER_TO_CLIENT = 1, -1
    # trace format: [[abstime, direction, size], ...]
    web_ip_sequence_dict, host_ip = extract_pcap(pcap_file)

    select_ip = list(web_ip_sequence_dict.keys())[0]
    # select the longest trace
    for web_ip, sequence in web_ip_sequence_dict.items():
        select_ip = web_ip if len(sequence) > len(web_ip_sequence_dict[select_ip]) else select_ip
    assert select_ip == "0.0.0.0"
    trace = []
    seq_seen = set()
    for ts, ip_src, ip_dst, seq, record_length in web_ip_sequence_dict[select_ip]:
        direction = CLIENT_TO_SERVER if ip_src == host_ip else SERVER_TO_CLIENT
        if seq not in seq_seen:
            seq_seen.add(seq)
            trace.append([ts, direction, record_length])
    return np.array(trace)


def run_task_func(__args):
    arg_dict = __args
    pcap_file, label_str = (arg_dict["pcap_file"], arg_dict["label"])
    feature = extract_pcap_client(pcap_file)
    return feature, label_str


print("Begin extract features")
# use multi-process to turn pcap to numpy

# multiprocessing to extract features
with multiprocessing.Pool(100) as pool:
    # results:list(tuple()) = pool.map(run_task_func, task_list)
    print("Begin multiprocessing to extract features")
    results: list(tuple()) = list(tqdm(pool.imap(run_task_func, task_list), total=len(task_list)))
# or use single process to extract features
# results = [run_task_func(task) for task in tqdm(task_list)]

results = [(feature, label_str) for feature, label_str in results if feature is not None]

traces, labels = zip(*results)

extracted_file = os.path.join("data", "extracted", dataset_name+"_filtered")
np.savez_compressed(extracted_file, traces=np.array(traces, dtype=object), labels=labels)

# %%
