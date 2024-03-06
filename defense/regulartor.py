import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.regulartor import RegularTorClientScheduleUnit, RegularTorServerScheduleUnit

from defense.base import Defense


class RegularTor(Defense):
    def __init__(self, param={}, mode="moderate", name="regulartor"):
        self.param = {
            "orig_rate": 277,
            "depreciation_rate": 0.94,
            "max_padding_budget": 3550,
            "burst_threshold": 3.55,
            "upload_ratio": 3.95,
            "delay_cap": 1.77,
        }
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        upload_ratio = self.param["upload_ratio"]
        delay_cap = self.param["delay_cap"]
        orig_rate = self.param["orig_rate"]
        depreciation_rate = self.param["depreciation_rate"]
        burst_threshold = self.param["burst_threshold"]

        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            RegularTorClientScheduleUnit(upload_ratio, delay_cap), DefensePlugin.TYPE_SCHEDULE
        )
        server.add_plugin(
            RegularTorServerScheduleUnit(orig_rate, depreciation_rate, burst_threshold),
            DefensePlugin.TYPE_SCHEDULE,
        )

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace
