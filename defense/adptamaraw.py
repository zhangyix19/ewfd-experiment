import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.adptamaraw import AdpTamarawScheduleUnit

from defense.base import Defense


class AdpTamaraw(Defense):
    def __init__(self, param={}, mode="moderate", name="adptamaraw"):
        self.param = {"tol": 0.8, "client_rate": 60, "server_rate": 150, "gap": (0.008, 0.04)}
        self.param.update(param)
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        tol = self.param["tol"]

        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            AdpTamarawScheduleUnit(self.param["tol"], self.param["client_rate"], self.param["gap"]),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            AdpTamarawScheduleUnit(self.param["tol"], self.param["server_rate"], self.param["gap"]),
            DefensePlugin.TYPE_SCHEDULE,
        )

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace
