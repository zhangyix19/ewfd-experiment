"""
captured [file tree]: pcaps, logs, and pngs
extracted [npz file]: contains packet traces(NDArray[Shape["* traces, * packets, [tick, dir, size] dims"], Float]) and labels(list[str])
truncated [npz file]: contains packet traces(NDArray[Shape["* traces, * packets, [tick, dir, size] dims"], Float]) and labels(list[str])
cell_level [npz file]: contains packet traces(NDArray[Shape["* traces, * cells, [tick, dir] dims"], Float]) and labels(NDArray[Shape["* labels"],Int])
defended [npz file]: contains packet traces(NDArray[Shape["* traces, * cells, [tick, dir] dims"], Float]) and labels(NDArray[Shape["* labels"],Int])
"""

import hashlib
import torch.multiprocessing as mp
import os
import time
from collections import Counter
from os.path import exists, join
from typing import *

import joblib
import numpy as np
import psutil
from nptyping import *
from tqdm import tqdm


np.set_printoptions(threshold=np.inf, suppress=True)


def pkt2cell_single_direction(trace_3, CELL_SIZE_WRAPPED_BY_TLS):
    CELL_SIZE = 514
    TLS_HEADER_LEN = CELL_SIZE_WRAPPED_BY_TLS - CELL_SIZE
    trace_1 = []
    buffer_size = 0
    new_tls_record = True
    for ts, __direction, pkt_size in trace_3:
        if (pkt_size - TLS_HEADER_LEN) % CELL_SIZE == 0:
            # exactly n cells
            # clear buffer if not empty, there should be a cell but not, so some bugs
            # however, because current packet is exactly n cells, so we let it go
            buffer_size = pkt_size
            new_tls_record = True
        else:
            # not exactly n cells, probably 1448, 1448, 1348 like
            # we assume the cell's ts is its first byte's ts
            buffer_size += pkt_size
        # we assume the cell's ts is its first byte's ts,
        # so when we see first byte, we push_back ts
        while buffer_size > CELL_SIZE / 2:
            if new_tls_record:
                buffer_size -= TLS_HEADER_LEN
                new_tls_record = False
            trace_1.append([ts])
            buffer_size -= CELL_SIZE

        if buffer_size == 0:
            new_tls_record = True
        # negative buffer_size means we preallocate a cell

    return np.array(trace_1)


def pkt2cell(trace_3, CELL_SIZE_WRAPPED_BY_TLS):
    c_trace3 = trace_3[np.where(trace_3[:, 1] > 0)]
    s_trace3 = trace_3[np.where(trace_3[:, 1] < 0)]

    c_trace1 = pkt2cell_single_direction(c_trace3, CELL_SIZE_WRAPPED_BY_TLS)
    s_trace1 = pkt2cell_single_direction(s_trace3, CELL_SIZE_WRAPPED_BY_TLS)

    if len(c_trace1) == 0:
        c_trace2 = np.zeros((0, 2))
    else:
        c_trace2 = np.concatenate((c_trace1, np.ones((c_trace1.shape[0], 1))), axis=1)
    if len(s_trace1) == 0:
        s_trace2 = np.zeros((0, 2))
    else:
        s_trace2 = np.concatenate((s_trace1, -np.ones((s_trace1.shape[0], 1))), axis=1)

    trace = np.concatenate((c_trace2, s_trace2), axis=0)

    return trace[trace[:, 0].argsort(kind="mergesort")]


def truncate(trace, PKT_THRESHOLD=2, START_THRESHOLD=10, END_THRESHOLD=12, PREFIX_THRESHOLD=100):
    start, end = 0, len(trace)
    t = trace[:, 0]

    pkt_diff = t[PKT_THRESHOLD:] - t[:-PKT_THRESHOLD]
    start_idxs = np.where(pkt_diff > START_THRESHOLD)[0]
    end_idxs = np.where(pkt_diff > END_THRESHOLD)[0]
    start_idxs = [0] + [idx + PKT_THRESHOLD for idx in start_idxs]
    end_idxs = [idx for idx in end_idxs] + [len(trace)]
    start, end = np.inf, -np.inf
    # a main range should start from a 10 sec interval and end before the first 15 sec interval

    for start_idx in start_idxs:
        for end_idx in end_idxs:
            if start_idx < end_idx:
                if end_idx - start_idx > end - start:
                    start, end = start_idx, end_idx
                break
    if end - start < 50:
        with open("debug.txt", "w") as f:
            print(trace.tolist(), file=f)
        print("start_idxs: ", start_idxs)
        print("end_idxs: ", end_idxs)
        print(
            f"start: {start}, end: {end}: "
            + str(end - start)
            + " packets left after truncation, too few, please check the trace\n total: "
            + str(len(trace))
        )

    return trace[start:end]


def func_wrapper(job):
    func, args = job
    return func(*args)


def run_parallel(task_name, func, argss: List[Tuple], proc=90):
    jobs = [(func, args) for args in argss]
    # with mp.Pool(int(psutil.cpu_count() * 0.1)) as p:
    with mp.Pool(proc) as p:
        imap_iter = p.imap(func_wrapper, jobs)
        results = [x for x in tqdm(imap_iter, total=len(jobs), desc=task_name)]
    return results


def cal_overhead(d, ud):
    ud = ud.copy()
    ud[:, 0] = ud[:, 0] - ud[0, 0]
    ud[:, 0] = (ud[:, 0] * 1000).astype(int) / 1000
    ud_time, d_time = ud[-1, 0] - ud[0, 0], d[-1, 0] - d[0, 0]
    time_overhead = d_time / ud_time - 1
    ud_bandwidth, d_bandwidth = (
        ud.shape[0] + 4,
        d.shape[0],
    )  # + 4 for 2 BEGIN and 2 END cells added
    bandwidth_overhead = d_bandwidth / ud_bandwidth - 1
    return [ud_time, time_overhead, ud_bandwidth, bandwidth_overhead]


def my_hash(*args):
    return hashlib.md5(str(args).encode()).hexdigest()


class TraceDataset:
    def __init__(
        self,
        name,
        data_dir="data",
        scenario="closed-world",
        cw_size=(100, 100),
        ow_size=(10000, 1),
        cell_size_wrapped_by_tls=536,
        do_truncate=True,
        use_cache=False,
    ):
        self.name = name
        self.data_dir = data_dir
        assert exists(self.data_dir), "data_dir does not exist"
        assert scenario in ["closed-world", "open-world"]
        self.scenario = scenario
        self.cw_size = cw_size
        self.ow_size = ow_size
        self.do_truncate = do_truncate
        self.cell_size_wrapped_by_tls = cell_size_wrapped_by_tls
        self.status = "unload"
        self.use_cache = use_cache
        self.init_hash()

        self.traces = []
        self.ud_traces = []
        self.labels = []  # str
        self.labels_int = []  # int

        self.truncated_dir = join(self.data_dir, "truncated")
        self.truncated_file = join(self.truncated_dir, self.name + ".npz")
        self.cell_level_dir = join(self.data_dir, "cell_level")
        self.cell_level_file = join(self.cell_level_dir, self.name + ".npz")
        self.defended_dir = join(self.data_dir, "defended", self.name)
        self.cache_dir = join(self.data_dir, "cache")

        self.prepared = False
        self.cw_label_map: Dict[str, int] = {}
        self.cw_index: List[int] = []
        self.ow_label_map: Dict[str, int] = {}
        self.ow_index: List[int] = []

        self.overheads = []  # overheads for each trace
        self.overhead = []  # [time_overhead, bandwidth_overhead]

    def init_hash(self):
        self.status = "unload"
        self.hash = my_hash(self.name, self.cell_size_wrapped_by_tls)

    def update_hash(self, status, *args):
        self.status = status
        self.hash = my_hash(self.hash, *args)

    def get_hash(self):
        return f"{self.status}_{self.hash}"

    def prepare_map(self):
        assert not self.use_cache
        if self.prepared:
            return
        assert len(self.traces) and len(self.labels)
        counter = Counter(self.labels)
        assert (
            counter.most_common(self.cw_size[0])[-1][1] >= self.cw_size[1]
        ), "There is not enough data for closed world"
        self.cw_label_map = {
            label: int_label
            for int_label, (label, _) in enumerate(counter.most_common(self.cw_size[0]))
        }
        cw_index = [list() for _ in range(self.cw_size[0])]
        for idx, label in enumerate(self.labels):
            if (
                label in self.cw_label_map
                and len(cw_index[self.cw_label_map[label]]) < self.cw_size[1]
            ):
                cw_index[self.cw_label_map[label]].append(idx)
        self.cw_index = [item for sublist in cw_index for item in sublist]
        assert len(self.cw_index) == self.cw_size[0] * self.cw_size[1]

        if self.scenario == "open-world":
            assert len(counter.most_common()) >= self.cw_size[0] + self.ow_size[0], (
                "There is not enough data for open world"
                + str(len(counter.most_common()))
                + " "
                + str(self.cw_size[0] + self.ow_size[0])
            )
            self.ow_label_map = {
                label: int_label
                for int_label, (label, _) in enumerate(
                    counter.most_common()[self.cw_size[0] : self.cw_size[0] + self.ow_size[0]]
                )
            }
            ow_index = [list() for _ in range(self.ow_size[0])]
            for idx, label in enumerate(self.labels):
                if (
                    label in self.ow_label_map
                    and len(ow_index[self.ow_label_map[label]]) < self.ow_size[1]
                ):
                    ow_index[self.ow_label_map[label]].append(idx)
            self.ow_index = [item for sublist in ow_index for item in sublist]
            assert len(self.ow_index) == self.ow_size[0] * self.ow_size[1]
        self.prepared = True
        return

    def __len__(self):
        if self.scenario == "closed-world":
            return self.cw_size[0] * self.cw_size[1]
        elif self.scenario == "open-world":
            return self.cw_size[0] * self.cw_size[1] + self.ow_size[0] * self.ow_size[1]
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        assert not self.use_cache
        self.prepare_map()
        if self.scenario == "closed-world":
            if isinstance(item, slice):
                map_item = self.cw_index[item]
                return [self.traces[idx] for idx in map_item], [
                    self.cw_label_map[self.labels[idx]] for idx in map_item
                ]
            elif isinstance(item, (list, np.ndarray)):
                map_item = [self.cw_index[int(idx)] for idx in item]
                return [self.traces[idx] for idx in map_item], [
                    self.cw_label_map[self.labels[idx]] for idx in map_item
                ]
            else:
                map_item = self.cw_index[item]
                return self.traces[map_item], self.cw_label_map[self.labels[map_item]]
        else:  # open world
            cw_len = len(self.cw_index)
            ow_len = len(self.ow_index)
            ow_label = self.cw_size[0]
            if isinstance(item, slice):
                slice_item = item
                traces = []
                labels = []
                for idx in range(cw_len + ow_len)[slice_item]:
                    if idx < cw_len:
                        traces.append(self.traces[self.cw_index[idx]])
                        labels.append(self.cw_label_map[self.labels[self.cw_index[idx]]])
                    else:
                        traces.append(self.traces[self.ow_index[idx - cw_len]])
                        labels.append(ow_label)
                return traces, labels
            elif isinstance(item, (list, np.ndarray)):
                idx_array = item
                traces = []
                labels = []
                for idx in idx_array:
                    idx = int(idx)
                    if idx < cw_len:
                        traces.append(self.traces[self.cw_index[idx]])
                        labels.append(self.cw_label_map[self.labels[self.cw_index[idx]]])
                    else:
                        traces.append(self.traces[self.ow_index[idx - cw_len]])
                        labels.append(ow_label)
                return traces, labels
            else:
                if item < cw_len:
                    return (
                        self.traces[self.cw_index[item]],
                        self.cw_label_map[str(self.labels[self.cw_index[item]])],
                    )
                else:
                    return self.traces[self.ow_index[item - cw_len]], ow_label

    def num_classes(self):
        if self.scenario == "closed-world":
            return self.cw_size[0]
        else:
            return self.cw_size[0] + 1

    def monitored_num(self):
        return self.cw_size[0]

    def monitored_labels(self):
        return list(self.cw_label_map.values())

    def unmonitored_label(self):
        return self.cw_size[0]

    def load_extracted(self):
        self.prepared = False
        extracted_dir = join(self.data_dir, "extracted")
        extracted_file = join(extracted_dir, self.name + ".npz")
        assert exists(extracted_file), f"Extracted file does not exist: {extracted_file}"
        data = np.load(extracted_file, allow_pickle=True)
        if "traces" in data.files:
            traces = data["traces"]
        else:
            traces = data["features"]
        labels = data["labels"]
        self.traces = [trace[trace[:, 1] != 0] for trace in traces]
        self.labels = labels

    def truncate(self):
        self.load_extracted()
        args = [(trace,) for trace in self.traces]
        self.traces = run_parallel("truncate", truncate, args)
        os.makedirs(self.truncated_dir, exist_ok=True)
        print(f"Saving truncate file to {self.truncated_file}")
        np.savez_compressed(
            self.truncated_file,
            traces=np.array(self.traces, dtype=object),
            labels=self.labels,
        )

    def load_truncated(self):
        self.prepared = False
        print(f"Try to load truncated data from {self.truncated_file} if existed")
        if not self.do_truncate:
            print("skip truncate")
            self.load_extracted()
            return
        if not exists(self.truncated_file):
            self.truncate()
        data = np.load(self.truncated_file, allow_pickle=True)
        traces = data["traces"]
        labels = data["labels"]
        self.traces = traces
        self.labels = labels

    def get_defend_file(self, defense_name):
        self.init_hash()
        self.update_hash("cell_level", os.stat(self.cell_level_file).st_mtime)
        os.makedirs(self.defended_dir, exist_ok=True)
        return join(self.defended_dir, defense_name + f"_{self.get_hash()}.npz")

    def to_cell_level(self):
        self.load_truncated()
        # self.traces = [pkt2cell(trace) for trace in self.traces]
        args = [(trace, self.cell_size_wrapped_by_tls) for trace in self.traces]
        self.traces = run_parallel("to_cell_level", pkt2cell, args)
        for trace in self.traces:
            trace[:, 0] = trace[:, 0] - trace[0, 0]
        os.makedirs(self.cell_level_dir, exist_ok=True)
        print(f"Saving cell level file to {self.cell_level_file}")
        np.savez_compressed(
            self.cell_level_file,
            traces=np.array(self.traces, dtype=object),
            labels=self.labels,
        )

    def save_undefend_file(self):
        assert not self.use_cache
        undefend_file = self.get_defend_file("undefend")
        print(f"Saving undefend data to {undefend_file}")
        np.savez_compressed(
            undefend_file,
            traces=np.array(self.traces, dtype=object),
            labels=self.labels,
            param=np.array(None),
            overhead=[0, 0],
        )
        self.init_hash()

    def load_cell_level(self):
        self.prepared = False
        print(f"Try to load cell level data from {self.cell_level_file} if existed")
        if not exists(self.cell_level_file):
            self.to_cell_level()
            self.save_undefend_file()
        else:
            data = np.load(self.cell_level_file, allow_pickle=True)
            self.traces = data["traces"] if not self.use_cache else []
            self.labels = data["labels"]
        self.ud_traces = self.traces.copy()
        self.init_hash()
        self.update_hash("cell_level", os.stat(self.cell_level_file).st_mtime)

    def defend(self, defense, parallel=True):
        assert not self.use_cache
        if "defends_parallel" not in dir(defense):

            while psutil.cpu_count() - psutil.cpu_percent(interval=1) < psutil.cpu_count() * 0.3:
                time.sleep(1)

            defend_func = defense.defend
            if parallel:
                args = [(trace,) for trace in self.ud_traces]
                self.traces = run_parallel("defend-" + defense.name, defend_func, args, parallel)
            else:
                self.traces = [
                    defend_func(trace)
                    for trace in tqdm(self.ud_traces, desc="defend-" + defense.name)
                ]
        else:
            self.traces = defense.defends_parallel(self.ud_traces)
        self.overheads = []
        self.overhead = []
        self.summary_overhead()
        os.makedirs(self.defended_dir, exist_ok=True)
        defended_file = self.get_defend_file(defense.name)
        assert self.overhead, "overhead not summarized"
        print(f"Saving {defense.name} data to {defended_file}")
        np.savez_compressed(
            defended_file,
            traces=np.array(self.traces, dtype=object),
            labels=self.labels,
            param=defense.param,
            overhead=self.overhead,
        )

    def load_defended(self, defense, parallel=100):
        if self.ud_traces is None:
            self.load_cell_level()
            self.ud_traces = self.traces
        dname = defense if isinstance(defense, str) else defense.name
        defended_file = self.get_defend_file(dname)
        print(f"Try to load {dname} defended data from {defended_file} if existed")
        if not exists(defended_file):
            assert not isinstance(defense, str), defended_file
            self.defend(defense, parallel)
        else:

            data = np.load(defended_file, allow_pickle=True)
            self.traces = data["traces"] if not self.use_cache else []
            self.labels = data["labels"]
            assert self.use_cache or len(self.traces) == len(self.ud_traces)
            self.overhead = self.read_overhead(dname)
        self.init_hash()
        self.update_hash(dname, os.stat(defended_file).st_mtime)
        return dname

    def read_overhead(self, defense):
        dname = defense if isinstance(defense, str) else defense.name
        defended_file = self.get_defend_file(dname)
        assert exists(defended_file), f"Defended file does not exist: {defended_file}"
        data = np.load(defended_file, allow_pickle=True)
        return list(data["overhead"]) if "overhead" in data.files else []

    def cal_overhead(self, defense=None):
        if defense:
            self.load_defended(defense)
        assert self.traces is not None and self.ud_traces is not None
        self.prepare_map()
        self.overheads = []
        idx_total = (
            self.cw_index if self.scenario == "closed-world" else self.cw_index + self.ow_index
        )
        # args = [(self.traces[idx], self.ud_traces[idx]) for idx in idx_total]
        # self.overhead = run_parallel("overhead", cal_overhead, args)
        self.overheads = [cal_overhead(self.traces[idx], self.ud_traces[idx]) for idx in idx_total]

    def summary_overhead(self, defense=None, format=None):
        if defense is not None:
            self.overhead = self.read_overhead(defense)

        if not self.overhead:
            self.cal_overhead(defense)
            time, time_overhead, bw, bw_overhead = np.array(self.overheads).T
            self.overhead = [
                np.average(time_overhead, weights=time),
                np.average(bw_overhead, weights=bw),
            ]

        if format == "str":
            dname = defense if isinstance(defense, str) else defense.name
            return f"{dname},{self.overhead[0]},{self.overhead[1]}"
        return self.overhead

    def to_wang_format(self, defense="undefend"):
        name = self.load_defended(defense)
        self.prepare_map()
        wang_dir = join(self.data_dir, "wang", self.name, name)
        os.makedirs(wang_dir, exist_ok=True)
        cw_counter = Counter()
        for idx in tqdm(self.cw_index, desc="closed-world"):
            trace = self.traces[idx]
            trace[:, 0] = trace[:, 0] - trace[0, 0]
            label = self.cw_label_map[self.labels[idx]]
            with open(join(wang_dir, f"{label}-{cw_counter[label]}"), "w") as f:
                for cell in trace:
                    f.write(f"{round(float(cell[0]),3)}\t{int(cell[1])}\n")
            cw_counter.update([label])
        if self.scenario == "open-world":
            for idx in tqdm(self.ow_index, desc="open-world"):
                trace = self.traces[idx]
                trace[:, 0] = trace[:, 0] - trace[0, 0]
                label = self.ow_label_map[self.labels[idx]]
                with open(join(wang_dir, f"{label}"), "w") as f:
                    for cell in trace:
                        f.write(f"{round(float(cell[0]),3)}\t{int(cell[1])}\n")

    def get_cached_data(self, attack):
        cache_path = join(
            self.cache_dir,
            f"{self.name}_{attack.name}_{self.status}_{my_hash(self.hash,self.scenario,self.cw_size if self.scenario=='closed-world' else self.ow_size)}.pkl",
        )
        print(f"Try to load cached processed data from {cache_path} if existed")
        if os.path.exists(cache_path):
            return joblib.load(cache_path)
        else:
            assert not self.use_cache
            data = attack.data_preprocess(*self[:])
            print(f"Saving processed data to {cache_path}")
            joblib.dump(data, cache_path)
            return data


def get_ds(name="ours", scenario="closed-world", use_cache=False):
    if name == "ssdf":
        ds = TraceDataset(
            name,
            scenario=scenario,
            cw_size=(95, 5),
            cell_size_wrapped_by_tls=543,
            use_cache=use_cache,
        )
    elif name == "smalldf":
        ds = TraceDataset(
            name,
            scenario=scenario,
            cw_size=(95, 100),
            cell_size_wrapped_by_tls=543,
            use_cache=use_cache,
        )
    elif name == "sdfow":
        ds = TraceDataset(
            name,
            scenario=scenario,
            cw_size=(95, 100),
            ow_size=(20000, 1),
            cell_size_wrapped_by_tls=543,
            do_truncate=False,
            use_cache=use_cache,
        )
    elif "df" in name:
        ds = TraceDataset(
            name,
            scenario=scenario,
            cw_size=(95, 1000),
            ow_size=(40000, 1),
            cell_size_wrapped_by_tls=543,
            do_truncate=False,
            use_cache=use_cache,
        )
    elif name == "sours":
        ds = TraceDataset(
            name,
            scenario=scenario,
            cell_size_wrapped_by_tls=536,
            use_cache=use_cache,
            cw_size=(100, 5),
        )
    else:
        ds = TraceDataset(
            name,
            scenario=scenario,
            cell_size_wrapped_by_tls=536,
            use_cache=use_cache,
        )
    ds.load_cell_level()
    return ds
