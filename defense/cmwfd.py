import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.cmwfd import CmwfdClientScheduleUnit, CmwfdServerScheduleUnit

from defense.base import Defense


class Cmwfd(Defense):
    def __init__(self, param={}, mode="moderate", name="cmwfd"):
        self.param = {
            "client": {
                "defenses": {
                    "random": {
                        "tamaraw": 0.2,
                        "regulartor": 0.2,
                        "adptamaraw": 0.2,
                        "front": 0.2,
                    },
                    "paired": {
                        "wfgan": 0.2,
                    },
                }
            },
            "server": {
                "defenses": {
                    "random": {
                        "tamaraw": 0.2,
                        "regulartor": 0.2,
                        "adptamaraw": 0.2,
                        "front": 0.2,
                    },
                    "paired": {
                        "wfgan": 0.2,
                    },
                }
            },
        }
        self.param.update(param)
        self.name = f"{name}_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(CmwfdClientScheduleUnit(self.param), DefensePlugin.TYPE_SCHEDULE)
        server.add_plugin(CmwfdServerScheduleUnit(self.param), DefensePlugin.TYPE_SCHEDULE)

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace
