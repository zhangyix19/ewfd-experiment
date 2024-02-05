import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.tamaraw import TamarawPaddingUnit, TamarawScheduleUnit

from defense.base import Defense


class Tamaraw(Defense):
    def __init__(self, param={}, mode="pessimistic", name="tamaraw"):
        self.param = {"client_interval": 0.04, "server_interval": 0.012}
        self.param.update(param)
        self.name = f"{name}_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        client_interval = self.param["client_interval"]
        server_interval = self.param["server_interval"]

        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(TamarawScheduleUnit(1, client_interval), DefensePlugin.TYPE_SCHEDULE)
        server.add_plugin(TamarawScheduleUnit(1, server_interval), DefensePlugin.TYPE_SCHEDULE)

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace
