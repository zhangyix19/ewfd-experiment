import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate

from defense.base import EWFDDefense
from ewfd_def.ezdef import (
    EzpaddingScheduleUnit,
    EzfixedScheduleUnit,
    EzlinearScheduleUnit,
    EzfixedrateScheduleUnit,
)


class Ezpadding(EWFDDefense):
    def __init__(self, param={}, name="ezpadding", mode="moderate"):
        super().__init__(name, mode)
        self.param = {
            "client_burst_num": 800,
            "server_burst_num": 1600,
            "client_budget": 1400,
            "server_budget": 2800,
            "interval": 8,
            "sigma": 1,
        }
        self.param.update(param)

    def defend_ewfd(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            EzpaddingScheduleUnit(
                self.param["client_burst_num"],
                self.param["client_budget"],
                self.param["interval"],
                self.param["sigma"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            EzpaddingScheduleUnit(
                self.param["server_burst_num"],
                self.param["server_budget"],
                self.param["interval"],
                self.param["sigma"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )

        return simulate(client, server, trace, self.mode)


class Ezfixed(EWFDDefense):
    def __init__(self, param={}, name="ezfixed", mode="moderate"):
        super().__init__(name, mode)
        self.param = {
            "client_base_rate": 5,
            "server_base_rate": 10,
            "burst_gap": 0.02,
            "wnd": 0.5,
            "flush_rate": 1.5,
        }
        self.param.update(param)

    def defend_ewfd(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            EzfixedScheduleUnit(
                self.param["client_base_rate"],
                self.param["burst_gap"],
                self.param["wnd"],
                self.param["flush_rate"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            EzfixedScheduleUnit(
                self.param["server_base_rate"],
                self.param["burst_gap"],
                self.param["wnd"],
                self.param["flush_rate"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )

        return simulate(client, server, trace, self.mode)


class Ezfixedrate(EWFDDefense):
    def __init__(self, param={}, name="ezfixedrate", mode="moderate"):
        super().__init__(name, mode)
        self.param = {"burst_size": 30}
        self.param.update(param)

    def defend_ewfd(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            EzfixedrateScheduleUnit(
                wnd=1,
                burst_gap=0.01,
                slow_rate=30,
                fast_rate=80,
                idle_threshold=10,
                slow_trigger=40,
                fast_trigger=50,
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            EzfixedrateScheduleUnit(
                wnd=1,
                burst_gap=0.01,
                slow_rate=100,
                fast_rate=500,
                idle_threshold=20,
                slow_trigger=180,
                fast_trigger=200,
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        return simulate(client, server, trace, self.mode)


class Ezlinear(EWFDDefense):
    def __init__(self, param={}, name="ezlinear", mode="moderate"):
        super().__init__(name, mode)
        self.param = {
            "init_rate": 40,
            "client_gradient": 80,
            "server_gradient": 240,
            "wnd": 0.5,
            "slow_down_factor": 0.5,
            "budget_rate": 0.5,
        }
        self.param.update(param)

    def defend_ewfd(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            EzlinearScheduleUnit(
                self.param["init_rate"],
                self.param["client_gradient"],
                self.param["wnd"],
                self.param["slow_down_factor"],
                self.param["budget_rate"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            EzlinearScheduleUnit(
                self.param["init_rate"],
                self.param["server_gradient"],
                self.param["wnd"],
                self.param["slow_down_factor"],
                self.param["budget_rate"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )

        return simulate(client, server, trace, self.mode)
