from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.regulartor import RegularTorClientScheduleUnit, RegularTorServerScheduleUnit

from defense.base import EWFDDefense


class RegularTor(EWFDDefense):
    def __init__(self, param={}, name="regulartor", mode="moderate"):
        super().__init__(name, mode)
        self.param = {
            "orig_rate": 277,
            "depreciation_rate": 0.94,
            "max_padding_budget": 3550,
            "burst_threshold": 3.55,
            "upload_ratio": 3.95,
            "delay_cap": 1.77,
        }

    def defend_ewfd(self, trace):
        upload_ratio = self.param["upload_ratio"]
        delay_cap = self.param["delay_cap"]
        orig_rate = self.param["orig_rate"]
        depreciation_rate = self.param["depreciation_rate"]
        burst_threshold = self.param["burst_threshold"]

        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            RegularTorClientScheduleUnit(upload_ratio, delay_cap), DefensePlugin.TYPE_SCHEDULE
        )
        server.add_plugin(
            RegularTorServerScheduleUnit(orig_rate, depreciation_rate, burst_threshold),
            DefensePlugin.TYPE_SCHEDULE,
        )

        return simulate(client, server, trace, self.mode)
