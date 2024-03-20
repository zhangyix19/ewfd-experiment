import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.front import FrontScheduleUnit

from defense.base import EWFDDefense


class Front(EWFDDefense):
    def __init__(self, param={}, mode="moderate", name="front"):
        super().__init__(name, mode)
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

    def defend_ewfd(self, trace):
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

        return simulate(client, server, trace, self.mode)

    def getTimestamps(self, wnd, num):
        # timestamps = sorted(np.random.exponential(wnd/2.0, num))
        # print(wnd, num)
        # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
        timestamps = sorted(np.random.rayleigh(wnd, num))
        # print(timestamps[:5])
        # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
        return np.reshape(timestamps, (len(timestamps), 1))
