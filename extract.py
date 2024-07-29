# %%
import argparse
import multiprocessing
import os
import socket

import dpkt
import numpy as np
from tqdm import tqdm

import imgvalid.validator as imgvalidator

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_name", type=str, default="undefend", help="dataset name")
parser.add_argument("-g", "--cuda_id", type=str, default="0", help="gpu id")
args = parser.parse_args()

dataset_name = args.dataset_name
cuda_id = args.cuda_id

# dataset_name = "undefend"
# cuda_id = 0


# get path list
def get_path_list(dataset_name: str):
    print("Begin get path list")
    captured_dir = os.path.join("data", "captured", dataset_name)
    paths = os.walk(captured_dir, followlinks=True)

    path_list = []
    for path, dir_lst, file_lst in paths:
        # should be a url dir
        if not "url" in os.path.basename(path):
            continue
        # should have label file
        label_file = os.path.join(path, "label")
        if not os.path.exists(label_file):
            continue
        with open(label_file) as f:
            label = f.readline()
            label = label.strip("\n")
            label = label.replace("http://", "")
            label = label.replace("https://", "")
        # should have image file
        captured_img_file = os.path.join(path, label + ".png")
        if not os.path.exists(captured_img_file):
            continue
        # should have pcap file
        pcap_file = os.path.join(path, "tcp.pcap")
        if not os.path.exists(pcap_file):
            continue
        # should have ip file
        ip_file = os.path.join(path, "../../ip")
        if not os.path.exists(ip_file):
            continue
        with open(ip_file) as f:
            client_in_ip = f.readline()
            client_in_ip = client_in_ip.strip("\n")
        # maybe have time file
        time_file = os.path.join(path, "time")
        if os.path.exists(time_file):
            with open(time_file) as f:
                time = f.readlines()
                time = [t.strip("\n") for t in time]
        else:
            time = [None, None]
        path_list.append(
            {
                "path": path,
                "captured_img_file": captured_img_file,
                "pcap_file": pcap_file,
                "client_in_ip": client_in_ip,
                "start_time": float(time[0]) if time[0] is not None else None,
                "end_time": float(time[1]) if time[1] is not None else None,
                "label": label,
            }
        )
    print("Finish get path list")
    return path_list


path_list = get_path_list(dataset_name)
print("path_list length: {}".format(len(path_list)))
# validate images
v = imgvalidator.Validator(device=f"cuda:{cuda_id}" if cuda_id != "cpu" else "cpu")
img_file_list = [path["captured_img_file"] for path in path_list]
valid_list = v.validate(img_file_list)
task_list = [path_list[i] for i in range(len(valid_list)) if valid_list[i]]


def extract_pcap(pcap_file: str, host_ip: str = None):
    __decoder = {
        dpkt.pcap.DLT_LOOP: dpkt.loopback.Loopback,
        dpkt.pcap.DLT_NULL: dpkt.loopback.Loopback,
        dpkt.pcap.DLT_EN10MB: dpkt.ethernet.Ethernet,
        dpkt.pcap.DLT_LINUX_SLL: dpkt.sll.SLL,
    }
    web_ip_sequence_dict = {}
    if host_ip is None:
        # detect host ip
        with open(pcap_file, "rb") as fin:
            pcapin = dpkt.pcap.Reader(fin)
            decode = __decoder[pcapin.datalink()]
            src_list = []
            dst_list = []
            while True:
                try:
                    ts, buf = next(pcapin)
                    pkt = decode(buf)
                    # get ip packet
                    ip = pkt.data
                    if not ip.p == dpkt.ip.IP_PROTO_TCP:
                        continue
                    # get tcp packet
                    tcp = ip.data
                    if not isinstance(tcp, dpkt.tcp.TCP):
                        continue
                    src_list.append(socket.inet_ntoa(ip.src))
                    dst_list.append(socket.inet_ntoa(ip.dst))
                    if len(set(src_list)) > 2 and len(set(dst_list)) > 2:
                        flag = True
                        for i in range(len(src_list)):
                            flag = flag and (
                                src_list[i] == socket.inet_ntoa(ip.src)
                                or dst_list[i] == socket.inet_ntoa(ip.src)
                            )
                        if flag:
                            host_ip = socket.inet_ntoa(ip.src)
                            break
                        flag = True
                        for i in range(len(src_list)):
                            flag = flag and (
                                src_list[i] == socket.inet_ntoa(ip.dst)
                                or dst_list[i] == socket.inet_ntoa(ip.dst)
                            )
                        if flag:
                            host_ip = socket.inet_ntoa(ip.dst)
                            break
                except:
                    break
            if host_ip is None:
                print(src_list)
                print(dst_list)
            assert host_ip is not None, "host ip not found"

    with open(pcap_file, "rb") as fin:
        pcapin = dpkt.pcap.Reader(fin)
        decode = __decoder[pcapin.datalink()]
        while True:
            try:
                ts, buf = next(pcapin)
                pkt = decode(buf)
                # get ip packet
                ip = pkt.data
                if not ip.p == dpkt.ip.IP_PROTO_TCP:
                    continue
                # get tcp packet
                tcp = ip.data
                if not isinstance(tcp, dpkt.tcp.TCP):
                    continue
                record_length = len(tcp.data)
                if record_length == 0:
                    continue
                ip_set = {socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst)}
                assert host_ip in ip_set, "ip.src: {}, ip.dst: {}, host_ip: {}".format(
                    socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst), host_ip
                )
                (web_ip,) = ip_set - {host_ip}
                if not web_ip in web_ip_sequence_dict:
                    web_ip_sequence_dict[web_ip] = list()
                web_ip_sequence_dict[web_ip].append(
                    (ts, socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst), tcp.seq, record_length)
                )
            except AssertionError as e:
                print(e)
                exit(1)
            except:
                break
    return web_ip_sequence_dict, host_ip


def extract_pcap_client(pcap_file: str, client_in_ip: str = None):
    CLIENT_TO_SERVER, SERVER_TO_CLIENT = 1, -1
    # trace format: [[abstime, direction, size], ...]
    web_ip_sequence_dict, client_in_ip = extract_pcap(pcap_file, client_in_ip)

    select_ip = list(web_ip_sequence_dict.keys())[0]
    # select the longest trace
    for web_ip, sequence in web_ip_sequence_dict.items():
        select_ip = web_ip if len(sequence) > len(web_ip_sequence_dict[select_ip]) else select_ip
    trace = []
    seq_seen = set()
    for ts, ip_src, ip_dst, seq, record_length in web_ip_sequence_dict[select_ip]:
        direction = CLIENT_TO_SERVER if ip_src == client_in_ip else SERVER_TO_CLIENT
        if seq in seq_seen:
            continue
        trace.append([ts, direction, record_length])
        seq_seen.add(seq)
    return np.array(trace)


def run_task_func(__args):
    arg_dict = __args
    pcap_file, client_in_ip, label_str = (
        arg_dict["pcap_file"],
        arg_dict["client_in_ip"],
        arg_dict["label"],
    )
    feature = extract_pcap_client(pcap_file, client_in_ip=client_in_ip)
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

extracted_file = os.path.join("data", "extracted", dataset_name)
np.savez_compressed(extracted_file, traces=np.array(traces, dtype=object), labels=labels)

# %%
