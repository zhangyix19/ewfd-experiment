"""
captured [file tree]: pcaps, logs, and pngs
extracted [npz file]: contains packet traces(NDArray[Shape["* traces, * packets, [tick, dir, size] dims"], Float]) and labels(list[str])
truncated [npz file]: contains packet traces(NDArray[Shape["* traces, * packets, [tick, dir, size] dims"], Float]) and labels(list[str])
cell_level [npz file]: contains packet traces(NDArray[Shape["* traces, * cells, [tick, dir] dims"], Float]) and labels(NDArray[Shape["* labels"],Int])
defended [npz file]: contains packet traces(NDArray[Shape["* traces, * cells, [tick, dir] dims"], Float]) and labels(NDArray[Shape["* labels"],Int])
"""

import os
from os.path import join, exists
from typing import *
from nptyping import *
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from defense import Defense
import pandas as pd
from collections import Counter


def pkt2cell_single_direction(trace_3, cell_size):
    trace_1 = []
    buffer_size = 0
    for ts, __direction, pkt_size in trace_3:
        if pkt_size % cell_size == 0:
            # exactly n cells
            # clear buffer if not empty, there should be a cell but not, so some bugs
            # however, because current packet is exactly n cells, so we let it go
            buffer_size = pkt_size
        else:
            # not exactly n cells, probably 1448, 1448, 1348 like
            # we assume the cell's ts is its first byte's ts
            buffer_size += pkt_size
        # we assume the cell's ts is its first byte's ts,
        # so when we see first byte, we push_back ts
        while buffer_size > 0:
            trace_1.append([ts])
            buffer_size -= cell_size
        # negative buffer_size means we preallocate a cell
    return np.array(trace_1)


def pkt2cell(trace_3, cell_size):
    c_trace3 = trace_3[np.where(trace_3[:, 1] > 0)]
    s_trace3 = trace_3[np.where(trace_3[:, 1] < 0)]

    c_trace1 = pkt2cell_single_direction(c_trace3, cell_size)
    s_trace1 = pkt2cell_single_direction(s_trace3, cell_size)

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


def run_parallel(task_name, func, argss: List[Tuple]):
    jobs = [(func, args) for args in argss]
    with mp.Pool(42) as p:
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


class TraceDataset:
    def __init__(
        self, name, data_dir, scenario="closed-world", cw_size=(100, 100), ow_size=(10000, 1)
    ):
        self.name = name
        self.data_dir = data_dir
        assert exists(self.data_dir), "data_dir does not exist"
        assert scenario in ["closed-world", "open-world"]
        self.scenario = scenario
        self.cw_size = cw_size
        self.ow_size = ow_size

        self.traces: List[NDArray[Shape["* pkts, * dims"], Float]] = None
        self.ud_traces: List[NDArray[Shape["* pkts, [ts, dir, size] dims"], Float]] = None
        self.labels: NDArray[Shape["* labels"], Any] = None
        self.labels_int: NDArray[Shape["* labels"], Int] = None
        self.do_truncate = True
        self.do_bundle = True
        self.cell_size = 536

        self.truncated_dir = join(self.data_dir, "truncated")
        self.truncated_file = join(self.truncated_dir, self.name + ".npz")
        self.cell_level_dir = join(self.data_dir, "cell_level")
        self.cell_level_file = join(self.cell_level_dir, self.name + ".npz")
        self.defended_dir = join(self.data_dir, "defended", self.name)

        self.prepared = False
        self.cw_label_map: Dict[str, int] = None
        self.cw_index = []
        self.ow_label_map: Dict[str, int] = None
        self.ow_index = []
        self.total_index = None

    # def cut_trace(self, length):
    #     if self.traces is not None:
    #         self.traces = [trace[:length] for trace in self.traces]
    #         self.traces = [
    #             np.pad(trace, ((0, length - trace.shape[0]), (0, 0)), "constant")
    #             if trace.shape[0] < length
    #             else trace
    #             for trace in self.traces
    #         ]

    def load_extracted(self):
        self.prepared = False
        extracted_dir = join(self.data_dir, "extracted")
        extracted_file = join(extracted_dir, self.name + ".npz")
        assert exists(extracted_file), f"Extracted file does not exist: {extracted_file}"
        data = np.load(extracted_file, allow_pickle=True)
        if "traces" in data.files:
            traces: List[NDArray[Shape["* pkts, [ts, dir, size] dims"], Float]] = data["traces"]
        else:
            traces: List[NDArray[Shape["* pkts, [ts, dir, size] dims"], Float]] = data["features"]
        labels: NDArray[Shape["* labels"], String] = data["labels"]
        self.traces = [trace[trace[:, 1] != 0] for trace in traces]
        self.labels = labels

    def truncate(self):
        if not self.do_truncate:
            print("skip truncate")
            return
        self.load_extracted()
        args = [(trace,) for trace in self.traces]
        self.traces = run_parallel("truncate", truncate, args)
        os.makedirs(self.truncated_dir, exist_ok=True)
        print(self.labels.shape)
        np.savez_compressed(
            self.truncated_file,
            traces=np.array(self.traces, dtype=object),
            labels=self.labels,
        )

    def load_truncated(self):
        self.prepared = False
        if not exists(self.truncated_file):
            self.truncate()
        data = np.load(self.truncated_file, allow_pickle=True)
        traces: List[NDArray[Shape["* pkts, [ts, dir, size] dims"], Float]] = data["traces"]
        labels: NDArray[Shape["* labels"], String] = data["labels"]
        self.traces = traces
        self.labels = labels

        # def bundle(self):
        #     if not self.do_bundle:
        #         label_set = set(self.labels)
        #         self.label_map = {label: idx for idx, label in enumerate(label_set)}
        #         self.labels = np.array([self.label_map[label] for label in self.labels])
        #         return
        #     trace_dict = {key: [] for key in self.labels}
        #     for idx, label in enumerate(self.labels):
        #         trace = self.traces[idx]
        #         if trace[-1, 0] - trace[0, 0] > 10:
        #             trace_dict[label].append(trace)

        #     trace_dict = {
        #         label: trace_list for label, trace_list in trace_dict.items() if len(trace_list) > 95
        #     }
        #     print({label: len(trace_dict[label]) for label in trace_dict})
        #     traces, labels = [], []
        #     self.label_map = {label: idx for idx, label in enumerate(trace_dict)}
        #     for idx, label in enumerate(trace_dict):
        #         traces_of_label = trace_dict[label][:100]
        #         traces.extend(traces_of_label)
        #         labels.extend([idx] * len(traces_of_label))
        #     self.traces = traces
        #     self.labels = np.array(labels)

        # def load_bundled(self):
        # bundled_dir = join(self.data_dir, "bundled")
        # bundled_file = join(bundled_dir, self.name + ".npz")
        # if not exists(bundled_file):
        #     self.load_truncated()
        #     self.bundle()
        #     os.makedirs(bundled_dir, exist_ok=True)
        #     np.savez_compressed(
        #         bundled_file,
        #         traces=np.array(self.traces, dtype=object),
        #         labels=self.labels,
        #         label_map=self.label_map,
        #     )
        # data = np.load(bundled_file, allow_pickle=True)
        # traces: List[NDArray[Shape["* pkts, [ts, dir, size] dims"], Float]] = data["traces"]
        # labels: NDArray[Shape["* labels"], Int] = data["labels"]
        # label_map: Dict[str, int] = data["label_map"].item()
        # self.traces = traces
        # self.labels = labels
        # self.label_map = label_map

    def to_cell_level(self):
        self.load_truncated()
        # self.traces = [pkt2cell(trace) for trace in self.traces]
        args = [(trace, self.cell_size) for trace in self.traces]
        self.traces = run_parallel("to_cell_level", pkt2cell, args)
        os.makedirs(self.cell_level_dir, exist_ok=True)
        np.savez_compressed(
            self.cell_level_file,
            traces=np.array(self.traces, dtype=object),
            labels=self.labels,
        )

    def load_cell_level(self):
        self.prepared = False
        if not exists(self.cell_level_file):
            self.to_cell_level()
        data = np.load(self.cell_level_file, allow_pickle=True)
        traces: List[NDArray[Shape["* pkts, [ts, dir] dims"], Float]] = data["traces"]
        labels: NDArray[Shape["* labels"], Any] = data["labels"]
        self.traces = traces
        self.labels = labels

    def defend(self, defense):
        if "defends_parallel" not in dir(defense):
            # if available cpu<40,sleep
            import psutil, time

            while psutil.cpu_count() - psutil.cpu_percent(interval=1) < 35:
                time.sleep(1)

            defend_func = defense.defend
            args = [(trace,) for trace in self.traces]
            self.traces = run_parallel("defend-" + defense.name, defend_func, args)
        else:
            self.traces = defense.defends_parallel(self.traces)
        os.makedirs(self.defended_dir, exist_ok=True)
        defended_file = join(self.defended_dir, defense.name + ".npz")
        np.savez_compressed(
            defended_file,
            traces=np.array(self.traces, dtype=object),
            labels=self.labels,
            param=defense.param,
        )

    def load_defended_by_name(self, defense_name: str, length=None):
        self.prepared = False
        self.load_cell_level()
        self.ud_traces = self.traces
        if defense_name != "undefend":
            defended_file = join(self.defended_dir, defense_name + ".npz")
            assert exists(defended_file), f"Defended file does not exist: {defended_file}"
            data = np.load(defended_file, allow_pickle=True)
            self.traces = data["traces"]
            self.labels = data["labels"]
            assert len(self.traces) == len(self.ud_traces)

    def load_defended(self, defense: Defense):
        self.prepared = False
        self.load_cell_level()
        self.ud_traces = self.traces
        defended_file = join(self.defended_dir, defense.name + ".npz")
        if not exists(defended_file):
            self.defend(defense)
        data = np.load(defended_file, allow_pickle=True)
        self.traces = data["traces"]
        self.labels = data["labels"]
        assert len(self.traces) == len(self.ud_traces)

    def cal_overhead(self, defense=None):
        if defense:
            if isinstance(defense, str):
                self.load_defended_by_name(defense)
            else:
                self.load_defended(defense)
        assert (
            self.ud_traces is not None and self.traces is not None
        ), "Give a defense or load defended first"
        self.prepare_map()
        self.overhead = []
        idx_total = (
            self.cw_index if self.scenario == "closed-world" else self.cw_index + self.ow_index
        )
        args = [(self.traces[idx], self.ud_traces[idx]) for idx in idx_total]
        self.overhead = run_parallel("overhead", cal_overhead, args)

    def summary_overhead(self, defense=None, format=None):
        self.cal_overhead(defense)
        overhead = np.array(self.overhead)
        time, time_overhead = overhead[:, 0], overhead[:, 1]
        bw, bw_overhead = overhead[:, 2], overhead[:, 3]
        OH = pd.DataFrame(index=["TIME OH", "BW OH"], columns=["AVG", "TOTAL"])
        # time_overhead_avg = (np.average(time_overhead), np.average(time_overhead, weights=time))
        # bw_overhead_avg = (np.average(bw_overhead), np.average(bw_overhead, weights=bw))
        OH["AVG"] = [np.average(time_overhead), np.average(bw_overhead)]
        OH["TOTAL"] = [np.average(time_overhead, weights=time), np.average(bw_overhead, weights=bw)]
        if format == "str":
            return ",".join(
                [
                    defense.name,
                    str(np.average(time_overhead, weights=time)),
                    str(np.average(bw_overhead, weights=bw)),
                    str(np.average(time_overhead)),
                    str(np.average(bw_overhead)),
                ]
            )
        return OH

    def prepare_map(self):
        if self.prepared:
            return
        assert self.labels is not None, "Please load data first"
        counter = Counter(self.labels)
        assert (
            counter.most_common(self.cw_size[0])[-1][1] >= self.cw_size[1]
        ), "There is not enough data for closed world"
        self.cw_label_map = {
            label: int_label
            for int_label, (label, _) in enumerate(counter.most_common(self.cw_size[0]))
        }
        self.cw_index = [list() for _ in range(self.cw_size[0])]
        for idx, label in enumerate(self.labels):
            if (
                label in self.cw_label_map
                and len(self.cw_index[self.cw_label_map[label]]) < self.cw_size[1]
            ):
                self.cw_index[self.cw_label_map[label]].append(idx)
        self.cw_index = [item for sublist in self.cw_index for item in sublist]
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
            self.ow_index = [list() for _ in range(self.ow_size[0])]
            for idx, label in enumerate(self.labels):
                if (
                    label in self.ow_label_map
                    and len(self.ow_index[self.ow_label_map[label]]) < self.ow_size[1]
                ):
                    self.ow_index[self.ow_label_map[label]].append(idx)
            self.ow_index = [item for sublist in self.ow_index for item in sublist]
            assert len(self.ow_index) == self.ow_size[0] * self.ow_size[1]
        self.prepared = True
        return

    def __len__(self):
        if self.scenario == "closed-world":
            return self.cw_size[0] * self.cw_size[1]
        elif self.scenario == "open-world":
            return self.cw_size[0] * self.cw_size[1] + self.ow_size[0] * self.ow_size[1]

    def __getitem__(self, item):
        self.prepare_map()
        if self.scenario == "closed-world":
            if isinstance(item, slice):
                map_item = self.cw_index[item]
                return [self.traces[idx] for idx in map_item], [
                    self.cw_label_map[self.labels[idx]] for idx in map_item
                ]
            elif isinstance(item, (list, np.ndarray)):
                map_item = [self.cw_index[idx] for idx in item]
                return [self.traces[idx] for idx in map_item], [
                    self.cw_label_map[self.labels[idx]] for idx in map_item
                ]
            else:
                map_item = self.cw_index[item]
                return self.traces[map_item], self.cw_label_map[self.labels[map_item]]
        else:  # open world
            cw_len = len(self.cw_index)
            ow_len = len(self.ow_index)
            ow_label = self.cw_size[0] + 1
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
                        self.cw_label_map[self.labels[self.cw_index[item]]],
                    )
                else:
                    return self.traces[self.ow_index[item - cw_len]], ow_label

    def num_classes(self):
        return self.cw_size[0] if self.scenario == "closed-world" else (self.cw_size[0] + 1)

    def monitored_num(self):
        return self.cw_size[0]

    def monitored_labels(self):
        return list(self.cw_label_map.values())

    def unmonitored_labels(self):
        return [self.cw_size[0] + 1]

    def to_wang_format(self, defense="undefend"):
        if isinstance(defense, str):
            self.load_defended_by_name(defense)
            name = defense
        else:
            self.load_defended(defense)
            name = defense.name
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
