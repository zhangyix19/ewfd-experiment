import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.wfgan import WfganClientScheduleUnit, WfganServerScheduleUnit

from defense.base import Defense


class Wfgan(Defense):
    def __init__(self, param={}, mode="moderate", name="wfgan"):
        self.param = {"tol": 0.6}
        self.param.update(param)
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        tol = self.param["tol"]
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(WfganClientScheduleUnit(tol), DefensePlugin.TYPE_SCHEDULE)
        server.add_plugin(WfganServerScheduleUnit(tol), DefensePlugin.TYPE_SCHEDULE)

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace
