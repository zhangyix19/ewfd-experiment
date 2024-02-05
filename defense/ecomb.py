import numpy as np

from defense.base import Defense
from defense.front import Front
from defense.tamaraw import Tamaraw


class EComb(Defense):
    """
    eBPF Based Combination Defense
    """

    def __init__(self, param={}, name="ecomb"):
        self.param = {
            "front": {
                "client_dummy_pkt_num": 1000,
                "server_dummy_pkt_num": 1000,
                "min_wnd": 1,
                "max_wnd": 10,
                "start_padding_time": 0,
                "client_min_dummy_pkt_num": 100,
                "server_min_dummy_pkt_num": 100,
            },
            "tamaraw": {
                "client_interval": 0.04,
                "server_interval": 0.012,
            },
            "ecomb": {"front_window": 200, "tamaraw_window": 200},
        }
        self.name = name
        self.front = Front(param=self.param["front"], name="ecomb-front")
        self.tamaraw = Tamaraw(param=self.param["tamaraw"], name="ecomb-tamaraw")

    def defend_real(self, trace):
        f_trace, t_trace = None, None
        f_window = self.param["ecomb"]["front_window"]
        f_whist = trace[:, 0] - np.pad(trace[:, 0], (f_window, 0), "constant")[:-f_window]
        f_pkt_count = np.arange(len(f_whist))
        f_pkt_count[f_pkt_count > f_window] = f_window
        # f_whist is f_pkt_count pkts 's time
        f_rate = f_pkt_count / f_whist
        f_switch = np.where(f_rate > 100)[0]
        f_switch = f_switch[f_switch > f_window]
        if len(f_switch) == 0:
            return self.front.defend_real(trace)
        f_switch_idx = f_switch[0]
        switch_ts = trace[f_switch_idx][0]
        f_trace = self.front.defend_real(trace[:f_switch_idx])
        f_trace = f_trace[f_trace[:, 0] < switch_ts]
        t_trace = self.tamaraw.defend_real(trace[f_switch_idx:])
        t_trace = t_trace[t_trace[:, 0] >= switch_ts]
        assert f_trace[-1][0] < t_trace[0][0]
        return np.concatenate((f_trace, t_trace), axis=0)


class EComb1(Defense):
    def __init__(self, param={}, name="ecomb1"):
        self.param = {
            "front": {
                "client_dummy_pkt_num": 1500,
                "server_dummy_pkt_num": 1500,
                "min_wnd": 1,
                "max_wnd": 15,
                "start_padding_time": 0,
                "client_min_dummy_pkt_num": 1,
                "server_min_dummy_pkt_num": 1,
            },
            "tamaraw-fast": {
                "client_interval": 0.02,
                "server_interval": 0.005,
            },
            "tamaraw-slow": {
                "client_interval": 0.04,
                "server_interval": 0.02,
            },
            "ecomb": {"window": 0.5, "burst_shrehold": 250, "tail_shrehold": 75},
        }
        self.param.update(param)
        self.name = name
        self.front = Front(param=self.param["front"], name="ecomb-front")
        self.tamaraw_fast = Tamaraw(param=self.param["tamaraw-fast"], name="ecomb-tamaraw-fast")
        self.tamaraw_slow = Tamaraw(param=self.param["tamaraw-slow"], name="ecomb-tamaraw-slow")

    def defend_real(self, trace):
        window = self.param["ecomb"]["window"]
        burst_shrehold = self.param["ecomb"]["burst_shrehold"]
        tail_shrehold = self.param["ecomb"]["tail_shrehold"]
        # idx
        t = trace[:, 0]
        pkt_in_window = [t[(t <= now) & (t > now - window)].shape[0] for now in t]
        burst_idx, tail_idx = None, None
        for i, pkt_num in enumerate(pkt_in_window):
            if not burst_idx and pkt_num >= burst_shrehold * window:
                burst_idx = i
            if burst_idx and pkt_num < tail_shrehold * window:
                tail_idx = i
                break
        if burst_idx is None:
            return self.front.defend_real(trace)

        burst_ts = trace[burst_idx][0]
        f_trace = self.front.defend_real(trace[:burst_idx])
        f_trace = f_trace[f_trace[:, 0] < burst_ts]
        if tail_idx is None:
            t_trace = self.tamaraw_fast.defend_real(trace[burst_idx:])
            t_trace = t_trace[t_trace[:, 0] >= burst_ts]
            return np.concatenate((f_trace, t_trace), axis=0)
        burst_t_trace = self.tamaraw_fast.defend_real(trace[burst_idx:tail_idx])
        burst_t_trace = burst_t_trace[burst_t_trace[:, 0] >= burst_ts]
        tail_ts = burst_t_trace[-1][0]
        tail_real_trace = trace[tail_idx:]
        tail_real_trace[:, 0] = tail_real_trace[:, 0] - tail_real_trace[0, 0] + tail_ts
        tail_t_trace = self.tamaraw_slow.defend_real(tail_real_trace)
        tail_t_trace = tail_t_trace[tail_t_trace[:, 0] >= tail_ts]

        return np.concatenate((f_trace, burst_t_trace, tail_t_trace), axis=0)


class ECombF(Defense):
    def __init__(self, param={}, name="ecombf"):
        self.param = {
            "front": {
                "client_dummy_pkt_num": 1500,
                "server_dummy_pkt_num": 1500,
                "min_wnd": 1,
                "max_wnd": 15,
                "start_padding_time": 0,
                "client_min_dummy_pkt_num": 1,
                "server_min_dummy_pkt_num": 1,
            },
            "tamaraw-slow": {
                "client_interval": 0.04,
                "server_interval": 0.012,
                "nseg": 50,
            },
            "tamaraw-fast": {
                "client_interval": 0.015,
                "server_interval": 0.003,
                "nseg": 50,
            },
            "trigger": {
                "window": 1,
                "morethan": 250,
                "lessthan": 80,
            },
        }
        self.param.update(param)
        self.name = name
        self.front = Front(param=self.param["front"], name="ecomb-front")
        self.tamaraw_slow = Tamaraw(param=self.param["tamaraw-slow"], name="ecomb-tamaraw-slow")
        self.tamaraw_fast = Tamaraw(param=self.param["tamaraw-fast"], name="ecomb-tamaraw-fast")

    def defend_real(self, trace):
        window = self.param["trigger"]["window"]
        morethan = self.param["trigger"]["morethan"]
        morethan_trigger = {
            "mode": "morethan",
            "window": self.param["trigger"]["window"],
            "rate": self.param["trigger"]["morethan"],
        }
        lessthan_trigger = {
            "mode": "lessthan",
            "window": self.param["trigger"]["window"],
            "rate": self.param["trigger"]["lessthan"],
        }
        # idx
        t = trace[:, 0]
        pkt_in_window = [t[(t <= now) & (t > now - window)].shape[0] for now in t]
        burst_idx = None
        for i, pkt_num in enumerate(pkt_in_window):
            if not burst_idx and pkt_num >= morethan * window:
                burst_idx = i
        if burst_idx is None:
            return self.front.defend_real(trace)

        burst_ts = trace[burst_idx][0]
        head = self.front.defend_real(trace[:burst_idx])
        head = head[head[:, 0] < burst_ts]

        remain = trace[burst_idx:]
        burst, remain = self.tamaraw_fast.defend_real(remain, trigger=lessthan_trigger)
        tail = self.tamaraw_slow.defend_real(remain)
        return np.concatenate((head, burst, tail), axis=0)


class ECombT(Defense):
    def __init__(self, param={}, name="ecombt"):
        self.param = {
            "tamaraw-slow": {
                "client_interval": 0.04,
                "server_interval": 0.012,
                "nseg": 50,
            },
            "tamaraw-fast": {
                "client_interval": 0.015,
                "server_interval": 0.003,
                "nseg": 50,
            },
            "trigger": {
                "window": 1,
                "morethan": 250,
                "lessthan": 80,
            },
        }
        self.param.update(param)
        self.name = name
        self.tamaraw_slow = Tamaraw(param=self.param["tamaraw-slow"], name="ecomb-tamaraw-slow")
        self.tamaraw_fast = Tamaraw(param=self.param["tamaraw-fast"], name="ecomb-tamaraw-fast")
        self.defenses = [self.tamaraw_slow, self.tamaraw_fast]

    def defend_real(self, trace):
        morethan_trigger = {
            "mode": "morethan",
            "window": self.param["trigger"]["window"],
            "rate": self.param["trigger"]["morethan"],
        }
        lessthan_trigger = {
            "mode": "lessthan",
            "window": self.param["trigger"]["window"],
            "rate": self.param["trigger"]["lessthan"],
        }
        head, remain = self.tamaraw_slow.defend_real(trace, trigger=morethan_trigger)
        burst, remain = self.tamaraw_fast.defend_real(remain, trigger=lessthan_trigger)
        tail = self.tamaraw_slow.defend_real(remain)
        return np.concatenate((head, burst, tail), axis=0)


class ECombTS(Defense):
    def __init__(self, param={}, name="ecombts"):
        self.param = {
            "tamaraw-slow": {
                "client_interval": 0.04,
                "server_interval": 0.012,
                "nseg": 50,
            },
            "tamaraw-fast": {
                "client_interval": 0.015,
                "server_interval": 0.003,
                "nseg": 50,
            },
            "trigger": {
                "window": 1,
                "morethan": 250,
                "lessthan": 80,
            },
        }
        self.param.update(param)
        self.name = name
        self.tamaraw_slow = Tamaraw(param=self.param["tamaraw-slow"], name="ecomb-tamaraw-slow")
        self.tamaraw_fast = Tamaraw(param=self.param["tamaraw-fast"], name="ecomb-tamaraw-fast")
        self.defenses = [self.tamaraw_slow, self.tamaraw_fast]

    def defend_real(self, trace):
        morethan_trigger = {
            "mode": "morethan",
            "window": self.param["trigger"]["window"],
            "rate": self.param["trigger"]["morethan"],
        }
        lessthan_trigger = {
            "mode": "lessthan",
            "window": self.param["trigger"]["window"],
            "rate": self.param["trigger"]["lessthan"],
        }
        segs = []
        remain = trace.copy()
        while len(remain) > 0:
            slow, remain = self.tamaraw_slow.defend_real(remain, trigger=morethan_trigger)
            segs.append(slow)
            if len(remain) == 0:
                break
            fast, remain = self.tamaraw_fast.defend_real(remain, trigger=lessthan_trigger)
            segs.append(fast)
            if len(remain) == 0:
                break
        return np.concatenate(segs, axis=0)
