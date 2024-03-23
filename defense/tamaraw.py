from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.tamaraw import TamarawScheduleUnit

from base import EWFDDefense


class Tamaraw(EWFDDefense):
    def __init__(self, param={}, name="tamaraw", mode="moderate"):
        super().__init__(name, mode)
        self.param = {"client_interval": 0.04, "server_interval": 0.012}
        self.param.update(param)

    def defend_ewfd(self, trace):
        client_interval = self.param["client_interval"]
        server_interval = self.param["server_interval"]

        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(TamarawScheduleUnit(1, client_interval), DefensePlugin.TYPE_SCHEDULE)
        server.add_plugin(TamarawScheduleUnit(1, server_interval), DefensePlugin.TYPE_SCHEDULE)

        return simulate(client, server, trace, self.mode)
