import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.switch import SwitchClientScheduleUnit, SwitchServerScheduleUnit

from defense.base import Defense


class Switch(Defense):
    def __init__(self, param={}, mode="moderate", name="switch"):
        self.param = {
            "defenses": {
                "random": {"tamaraw": 1, "regulartor": 1, "front": 1},
                "paired": {"wfgan": 1},
            }
        }
        self.param.update(param)

        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(SwitchClientScheduleUnit(self.param), DefensePlugin.TYPE_SCHEDULE)
        server.add_plugin(SwitchServerScheduleUnit(self.param), DefensePlugin.TYPE_SCHEDULE)

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace
