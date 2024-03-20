import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.adptamaraw import AdpTamarawScheduleUnit

from defense.base import EWFDDefense


class AdpTamaraw(EWFDDefense):
    def __init__(self, param={}, mode="moderate", name="adptamaraw"):
        self.param = {"tol": 0.8, "client_rate": 60, "server_rate": 150, "gap": (0.008, 0.04)}
        self.param.update(param)

    def defend_ewfd(self, trace):
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

        return simulate(client, server, trace, self.mode)
