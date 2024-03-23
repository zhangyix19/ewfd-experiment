from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.wfgan import WfganClientScheduleUnit, WfganServerScheduleUnit

from base import EWFDDefense


class Wfgan(EWFDDefense):
    def __init__(self, param={}, name="wfgan", mode="moderate"):
        super().__init__(name, mode)
        self.param = {"tol": 0.6}
        self.param.update(param)

    def defend_ewfd(self, trace):
        tol = self.param["tol"]
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(WfganClientScheduleUnit(tol), DefensePlugin.TYPE_SCHEDULE)
        server.add_plugin(WfganServerScheduleUnit(tol), DefensePlugin.TYPE_SCHEDULE)

        return simulate(client, server, trace, self.mode)
