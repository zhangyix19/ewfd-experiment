import numpy as np
from tqdm import tqdm
import numpy as np
from ewfd_def.ewfd import DefensePlugin, TorOneDirection, simulate, ScheduleUnit


class EWFDDefense:
    def __init__(self, name, mode):
        self.param = {}
        self.name = name
        if mode != "moderate":
            self.name += f"_{mode}"
        self.mode = mode

    def defend_ewfd(self, trace):
        raise NotImplementedError

    def defend(self, trace):
        trace = trace.copy()
        trace[:, 0] = trace[:, 0] * 1000
        trace = trace.astype(int)
        defended_trace = self.defend_ewfd(trace)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace


class Empty(EWFDDefense):
    def __init__(self):
        super().__init__("empty", "moderate")

    def defend_ewfd(self, trace):
        client = TorOneDirection()
        server = TorOneDirection()
        return simulate(client, server, trace, self.mode)
