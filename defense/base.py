import numpy as np
from tqdm import tqdm


class Defense:
    DUMMY_PKT = 888
    DELAYED_PKT = 999
    name = "base"
    can_parallel = True

    def init(self, param):
        raise NotImplementedError

    @staticmethod
    def get_real(trace):
        return trace[abs(trace[:, 1]) == 1]

    def defend_real(self, trace):
        # 处理单条trace
        raise NotImplementedError

    def defend(self, trace):
        # 处理单条带0的trace
        length = trace.shape[0]
        real_trace = self.get_real(trace)
        # 相对时间
        real_trace[:, 0] = real_trace[:, 0] - real_trace[0, 0]
        return self.defend_real(real_trace)
