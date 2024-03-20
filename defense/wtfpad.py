import numpy as np
import ewfd_def.spring as spring
import ewfd_def.interspace as interspace
import ewfd_def.wtfpad as wtfpad

from defense.base import EWFDDefense


class Spring(EWFDDefense):
    def __init__(self, name="spring"):
        self.name = name

    def defend_ewfd(self, trace):
        new_column = np.full(len(trace), 536)
        trace = np.append(trace, new_column[:, np.newaxis], axis=1)
        trace = trace.astype(int)
        defended_trace = spring.run(trace)
        return defended_trace[:, :2]


class Interspace(EWFDDefense):
    def __init__(self, name="interspace"):
        self.name = name

    def defend_ewfd(self, trace):
        new_column = np.full(len(trace), 536)
        trace = np.append(trace, new_column[:, np.newaxis], axis=1)
        trace = trace.astype(int)
        defended_trace = interspace.run(trace)
        return defended_trace[:, :2]


class Wtfpad(EWFDDefense):
    def __init__(self, name="wtfpad"):
        self.name = name

    def defend_ewfd(self, trace):

        new_column = np.full(len(trace), 536)
        trace = np.append(trace, new_column[:, np.newaxis], axis=1)
        trace = trace.astype(int)
        defended_trace = wtfpad.Wtfpad().run(trace)
        return defended_trace[:, :2]
