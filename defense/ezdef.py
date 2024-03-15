import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate

from defense.base import Defense
from ewfd_def.ezdef import (
    EzpaddingScheduleUnit,
    EzfixedScheduleUnit,
    EzlinearScheduleUnit,
    EzfixedrateScheduleUnit,
)


class Ezpadding(Defense):
    def __init__(self, param={}, mode="moderate", name="ezpadding"):
        self.param = {
            "client_budget": 1500,
            "server_budget": 3000,
            "interval": 8,
            "sigma": 1,
        }
        self.param.update(param)
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_real(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            EzpaddingScheduleUnit(
                self.param["client_budget"], self.param["interval"], self.param["sigma"]
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            EzpaddingScheduleUnit(
                self.param["server_budget"], self.param["interval"], self.param["sigma"]
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace


class Ezfixed(Defense):
    def __init__(self, param={}, mode="moderate", name="ezfixed"):
        # base_rate=10, burst_gap=0.02, wnd=0.5, flush_rate=1.5
        self.param = {
            "client_base_rate": 5,
            "server_base_rate": 10,
            "burst_gap": 0.02,
            "wnd": 0.5,
            "flush_rate": 1.5,
        }
        self.param.update(param)
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_real(self, trace):
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

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace


class Ezfixedrate(Defense):
    def __init__(self, param={}, mode="moderate", name="ezfixedrate"):
        self.param = {"burst_size": 30}
        self.param.update(param)
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_real(self, trace):
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
        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace


class Ezlinear(Defense):
    def __init__(self, param={}, mode="moderate", name="ezlinear"):
        # init_rate=40, increase_gradient=200, wnd=0.5, slow_down_factor=0.5, budget_rate=0.5
        self.param = {
            "init_rate": 40,
            "client_gradient": 80,
            "server_gradient": 240,
            "wnd": 0.5,
            "slow_down_factor": 0.5,
            "budget_rate": 0.5,
        }
        self.param.update(param)
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_real(self, trace):
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

        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = simulate(client, server, trace, self.mode)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace
