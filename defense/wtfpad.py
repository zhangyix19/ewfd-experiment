import sys
import numpy as np
import random
from random import randint
from math import ceil
from bisect import bisect_right, insort_left
from defense.base import Defense

# Directions
IN = 1
OUT = -1

# State
WAIT = 0
BURST = 1
GAP = 2

# Action
SND = 0
RCV = 1

# Event
NULL_EVENT = 0
SEND_DUMMY = 1

# Mappings
EP2DIRS = {"client": OUT, "server": IN}
MODE2STATE = {"gap": GAP, "burst": BURST}
ON2ACTION = {"snd": SND, "rcv": RCV}

INF = float("inf")
# MPU
TOR_CELL_SIZE = 536


def create_exponential_bins(a=None, b=None, n=None):
    """
    Return a partition of the interval [a, b] with n number of bins.
    """
    return ([b] + [(b - a) / 2.0**k for k in range(1, n)] + [a])[::-1]


def init_distrib(
    name: str,
    param_str: str,
    num_samples: int = 10000,
    min_iat: float = 0,
    max_iat: float = 10,
    bin_size: int = 30,
) -> dict:
    dist, params = param_str.split(",", 1)
    inf_config, dist_params = params.split(",", 1)

    # TODO: add supported distribution if desired, it only contains norm
    if dist == "norm":
        mu, sigma = map(float, [x.strip() for x in dist_params.split(",")])
        counts, bins = np.histogram(
            [s for s in np.random.normal(mu, sigma, num_samples) if s > 0],
            bins=create_exponential_bins(a=min_iat, b=max_iat, n=bin_size),
        )

    else:
        raise ValueError("Unknown probability distribution.")

    d = dict(zip(list(bins) + [INF], [0] + list(counts) + [0]))

    # remove 0 inter-arrival times
    d[0] = 0

    # Setting the histograms' infinity bins.
    burst = int(inf_config.strip())
    other_toks = sum([v for k, v in d.items() if k != INF])

    if "gap" in name:
        d[INF] = ceil(other_toks / (burst - 1))
    elif "burst" in name:
        d[INF] = ceil(other_toks / burst)

    return d


class Histogram:
    """Provides methods to generate and sample histograms of prob distributions."""

    def __init__(self, name: str, hist: dict):
        self.name = name
        self.hist = hist

        # create template histogram
        self.template = dict(hist)

        # store labels in a list for fast search over keys
        self.labels = sorted(self.hist.keys())
        self.n = len(self.labels)

    def remove_token(self, f):
        if f in self.labels:
            label = f
        else:
            label = self.labels[bisect_right(self.labels, f)]

        # get labels of non-empty bins
        pos_counts = [label for label in self.labels if self.hist[label] > 0]

        # remove tokens from label
        # if there is none, continue removing tokens on the right.
        if label not in pos_counts:
            if label < pos_counts[0]:
                label = pos_counts[0]
            else:
                label = pos_counts[bisect_right(pos_counts, label) - 1]

        self.hist[label] -= 1

        # if histogram is empty, refill the histogram
        if sum(self.hist.values()) == 0:
            self.hist = dict(self.template)

    def random_sample(self) -> float:
        """Draw and return a sample from the histogram."""
        total_tokens = int(sum(self.hist.values()))
        prob = randint(1, total_tokens) if total_tokens > 0 else 0

        for i, label_i in enumerate(self.labels):
            prob -= self.hist[label_i]
            if prob > 0:
                continue
            if i == self.n - 1:
                return label_i
            label_i_1 = 0 if i == 0 else self.labels[i - 1]
            if label_i == INF:
                return INF
            p = (
                label_i + (label_i_1 - label_i) * random.random()
            )  # get a random in [label_i_1, label_i)
            return p

        raise ValueError("In `histo.random_sample`: probability is larger than range of counts!")


class Flow(object):
    # record state and event of the flow
    def __init__(self, direction: int):
        self.state = BURST
        self.direction = direction
        self.timeout = 0.0
        self.expired = False
        self.dummy = False


class Wtfpad(Defense):
    def __init__(self, param=None):
        if param is None:
            param = {}
        self.name = "wtfpad"
        self.param = {
            "dists": {  # the burst/gap distribution of send/receive in client/server, only support norm distribution
                "client_snd_burst_dist": "norm, 9, 0.001564159, 0.052329599",
                "client_snd_gap_dist": "norm, 21, 0.06129599, 0.03995375",
                "client_rcv_burst_dist": "norm, 9, 0.0000128746, 0.0009227229",
                "client_rcv_gap_dist": "norm, 21, 0.0001368523, 0.0009233190",
                "server_snd_burst_dist": "norm, 19, 0.00003600121, 0.02753485",
                "server_snd_gap_dist": "norm, 34, 0.01325997, 0.0973761",
                "server_rcv_burst_dist": "norm, 19, 0.000004053116, 0.01264329",
                "server_rcv_gap_dist": "norm, 34, 0.01325997, 0.0126454036",
            },
            "num_bins": 30,  # the number of bins
            "min_iat": 0,  # the minimum bin
            "max_iat": 10,  # the maximum bin except inf
            "num_samples": 10000,
        }
        self.param.update(param)

        self.hist = self.initialize_distributions()

    @staticmethod
    def update_state(flow: Flow) -> bool:
        """Switch state accordingly to AP machine state."""
        if flow.state == WAIT and not flow.dummy:
            flow.state = BURST
        elif flow.state == GAP and not flow.dummy:
            flow.state = BURST
        elif flow.state == BURST and (flow.expired or flow.dummy):
            flow.state = GAP
        elif flow.state == BURST and flow.timeout == INF:
            flow.state = WAIT
        elif flow.state == GAP and flow.timeout == INF:
            flow.state = BURST
        else:
            return False

        return True

    def initialize_distributions(self) -> dict:
        on = {"snd": None, "rcv": None}
        dirs = {IN: dict(on), OUT: dict(on)}
        hist = {BURST: dict(dirs), GAP: dict(dirs)}
        for k, v in self.param["dists"].items():
            endpoint, on, mode, _ = k.split("_")
            s = MODE2STATE[mode]
            d = EP2DIRS[endpoint]
            m = ON2ACTION[on]
            hist[s][d][m] = Histogram(
                k,
                init_distrib(
                    k,
                    v,
                    self.param["num_samples"],
                    self.param["min_iat"],
                    self.param["max_iat"],
                    self.param["num_bins"],
                ),
            )
        return hist

    def add_padding(self, i: int, trace, flow: Flow, on: int):
        """Generate a dummy packet."""
        packet = trace[i]

        if flow.state == WAIT:
            return False

        timeout = INF
        histogram = self.hist[flow.state][flow.direction][on]
        if histogram is not None:
            timeout = histogram.random_sample()

        # compute iat this packet and next packet
        current_packet = trace[i]
        next_packet = trace[-1]
        for p in trace[i + 1 :]:
            if p[1] == current_packet[1]:
                next_packet = p
                break
        iat = next_packet[0] - current_packet[0]

        if iat <= 0:
            return False

        if timeout < iat:
            # timeout has expired
            flow.expired, flow.timeout = True, timeout

            # the timeout has expired, we send a dummy packet
            dummy_ts = packet[0] + timeout
            dummy_len = TOR_CELL_SIZE
            dummy = [dummy_ts, flow.direction]

            iat = timeout  # correct the timeout

            # add dummy to trace
            insort_left(trace, dummy)

        # remove the token from histogram
        if histogram is not None:
            histogram.remove_token(iat)

        return True

    def defend_real(self, trace):
        # Here, the wtfpad simulation is quite different from the real one. Since the data is collected on the CLIENT
        # side, it is used as the SERVER side data by reversing the direction and ignoring the effect of link
        # transmission.

        flows = {IN: Flow(IN), OUT: Flow(OUT)}

        trace_l = trace.tolist()

        for i, packet in enumerate(trace_l):
            direction = packet[1]

            flow = flows[direction]
            oppflow = flows[-direction]  # opposite direction

            # update state
            self.update_state(flow)

            # run adaptive padding in the flow direction
            flow.dummy = self.add_padding(i, trace_l, flow, SND)

            # run adaptive padding in the opposite direction,
            # as if the packet was received at the other side
            oppflow.dummy = self.add_padding(i, trace_l, oppflow, RCV)

        # sort race by timestamp
        trace_l.sort(key=lambda x: x[0])

        return np.array(trace_l)
