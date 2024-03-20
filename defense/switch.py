from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.switch import SwitchClientScheduleUnit, SwitchServerScheduleUnit

from defense.base import EWFDDefense


class Switch(EWFDDefense):
    def __init__(self, param={}, name="switch", mode="moderate"):
        super().__init__(name, mode)
        self.param = {
            "defenses": {
                "random": {"tamaraw": 1, "regulartor": 1, "front": 1},
                "paired": {"wfgan": 1},
            },
            "params": {},
        }
        self.param.update(param)

    def defend_ewfd(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(SwitchClientScheduleUnit(self.param), DefensePlugin.TYPE_SCHEDULE)
        server.add_plugin(SwitchServerScheduleUnit(self.param), DefensePlugin.TYPE_SCHEDULE)

        return simulate(client, server, trace, self.mode)
