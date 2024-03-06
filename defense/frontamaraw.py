import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.tamaraw import TamarawPaddingUnit, TamarawScheduleUnit
from ewfd_def.front import FrontPaddingUnit, FrontScheduleUnit
from defense.base import Defense


class FronTamaraw(Defense):
    def __init__(self, param={}, mode="pessimistic", name="frontamaraw"):
        self.param = {
            "server_interval": 0.012,
            "client_dummy_pkt_num": 3000,
            "min_wnd": 1,
            "max_wnd": 13,
            "start_padding_time": 0,
            "client_min_dummy_pkt_num": 1,
        }
        self.param.update(param)
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
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
            TamarawScheduleUnit(1, self.param["server_interval"]),
            DefensePlugin.TYPE_SCHEDULE,
        )

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace
