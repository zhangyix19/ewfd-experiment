from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.tamaraw import TamarawScheduleUnit
from ewfd_def.front import FrontScheduleUnit
from base import EWFDDefense


class FronTamaraw(EWFDDefense):
    def __init__(self, param={}, name="frontamaraw", mode="pessimistic"):
        super().__init__(name, mode)
        self.param = {
            "server_interval": 0.012,
            "client_dummy_pkt_num": 3000,
            "min_wnd": 1,
            "max_wnd": 13,
            "start_padding_time": 0,
            "client_min_dummy_pkt_num": 1,
        }
        self.param.update(param)

    def defend_ewfd(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        client.add_plugin(
            FrontScheduleUnit(
                self.param["client_dummy_pkt_num"],
                self.param["min_wnd"],
                self.param["max_wnd"],
            ),
            DefensePlugin.TYPE_SCHEDULE,
        )
        server.add_plugin(
            TamarawScheduleUnit(1, self.param["server_interval"]),
            DefensePlugin.TYPE_SCHEDULE,
        )

        return simulate(client, server, trace, self.mode)
