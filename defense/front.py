import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.front import FrontPaddingUnit, FrontScheduleUnit

from defense.base import Defense


class Front(Defense):
    def __init__(self, param={}, mode="pessimistic", name="front"):
        self.param = {
            "client_dummy_pkt_num": 3000,
            "server_dummy_pkt_num": 3000,
            "min_wnd": 1,
            "max_wnd": 13,
            "start_padding_time": 0,
            "client_min_dummy_pkt_num": 1,
            "server_min_dummy_pkt_num": 1,
        }
        self.param.update(param)
        self.name = f"{name}_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            FrontScheduleUnit(
                self.param["client_dummy_pkt_num"],
                self.param["min_wnd"],
                self.param["max_wnd"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            FrontScheduleUnit(
                self.param["server_dummy_pkt_num"],
                self.param["min_wnd"],
                self.param["max_wnd"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace

    def getTimestamps(self, wnd, num):
        # timestamps = sorted(np.random.exponential(wnd/2.0, num))
        # print(wnd, num)
        # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
        timestamps = sorted(np.random.rayleigh(wnd, num))
        # print(timestamps[:5])
        # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
        return np.reshape(timestamps, (len(timestamps), 1))
