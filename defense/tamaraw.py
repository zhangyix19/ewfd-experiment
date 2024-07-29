from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.tamaraw import TorOneDirectionFixed

from base import EWFDDefense


class Tamaraw(EWFDDefense):
    def __init__(self, param={}, name="tamaraw", mode="moderate"):
        super().__init__(name, mode)
        self.param = {"client_interval": 0.04, "server_interval": 0.012}
        self.param.update(param)

    def defend_ewfd(self, trace):
        client = TorOneDirectionFixed(role="client", pace=1 / self.param["client_interval"])
        server = TorOneDirectionFixed(role="server", pace=1 / self.param["server_interval"])

        return simulate(client, server, trace, self.mode)
