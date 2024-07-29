from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate
from ewfd_def.pred import TorOneDirectionPred

from base import EWFDDefense


class Pred(EWFDDefense):
    def __init__(self, param={}, name="pred", mode="moderate"):
        super().__init__(name, mode)
        self.param = {
            "wnd": 0.5,
            "security": 0.5,
            "data_vs_time": 1,
            # "flex": 0,
            # "abs": 20,
            # "min": 10,
            "model": "tcn",
            "best": False,
        }
        self.param.update(param)

    def defend_ewfd(self, trace):
        client = TorOneDirectionPred(
            role="client",
            wnd=self.param["wnd"],
            predict_model=self.param["model"],
            security=self.param["security"],
            data_vs_time=self.param["data_vs_time"],
            pace_min=10,
            best=self.param["best"],
        )
        server = TorOneDirectionPred(
            role="server",
            wnd=self.param["wnd"],
            predict_model=self.param["model"],
            security=self.param["security"],
            data_vs_time=self.param["data_vs_time"],
            pace_min=20,
            best=self.param["best"],
        )

        return simulate(client, server, trace, self.mode)
