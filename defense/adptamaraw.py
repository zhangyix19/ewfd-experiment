import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.adptamaraw import AdpTamarawScheduleUnit

from defense.base import Defense


class AdpTamaraw(Defense):
    def __init__(self, param={}, mode="pessimistic", name="adptamaraw"):
        self.param = {
            "tol": 0.8,
            "client_burst_size": 3,
            "server_burst_size": 5,
            "client_interval": 3 * 0.046,
            "server_interval": 0.05,
        }
        self.param.update(param)
        self.name = f"{name}_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        tol = self.param["tol"]
        client_interval = self.param["client_interval"]
        server_interval = self.param["server_interval"]

        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            AdpTamarawScheduleUnit(
                self.param["tol"], self.param["client_burst_size"], self.param["client_interval"]
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            AdpTamarawScheduleUnit(
                self.param["tol"], self.param["server_burst_size"], self.param["server_interval"]
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
