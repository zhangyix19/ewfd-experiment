import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.customized import CustomizedScheduleUnit

from base import EWFDDefense


class Customized(EWFDDefense):
    def __init__(self, client_config, server_config, name, mode="moderate"):
        super().__init__("customized-" + name.removeprefix("customized-"), mode)
        self.client_config = client_config
        self.server_config = server_config

    def defend_ewfd(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            CustomizedScheduleUnit(self.client_config),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            CustomizedScheduleUnit(self.server_config),
            DefensePlugin.TYPE_SCHEDULE,
        )

        return simulate(client, server, trace, self.mode)

    def getTimestamps(self, wnd, num):
        # timestamps = sorted(np.random.exponential(wnd/2.0, num))
        # print(wnd, num)
        # timestamps = sorted(abs(np.random.normal(0, wnd, num)))
        timestamps = sorted(np.random.rayleigh(wnd, num))
        # print(timestamps[:5])
        # timestamps = np.fromiter(map(lambda x: x if x <= wnd else wnd, timestamps),dtype = float)
        return np.reshape(timestamps, (len(timestamps), 1))
