import numpy as np
import ewfd_def.spring as spring
import ewfd_def.interspace as interspace
import ewfd_def.wtfpad as wtfpad

from defense.base import Defense


class Spring(Defense):
    def __init__(self, name="spring"):
        self.name = name
        self.param = {}

    def defend_real(self, trace):
        org_trace = trace.copy()
        org_trace[:, 0] = org_trace[:, 0] * 1000
        org_trace = org_trace.astype(int)
        new_column = np.full(len(org_trace), 536)
        trace = np.append(org_trace, new_column[:, np.newaxis], axis=1)
        trace = trace.astype(int)
        defended_trace = spring.run(trace)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace[:, :2]


class Interspace(Defense):
    def __init__(self, name="interspace"):
        self.name = name
        self.param = {}

    def defend_real(self, trace):
        org_trace = trace.copy()
        org_trace[:, 0] = org_trace[:, 0] * 1000
        org_trace = org_trace.astype(int)
        new_column = np.full(len(org_trace), 536)
        trace = np.append(org_trace, new_column[:, np.newaxis], axis=1)
        trace = trace.astype(int)
        defended_trace = interspace.run(trace)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace[:, :2]


class Wtfpad(Defense):
    def __init__(self, name="wtfpad"):
        self.name = name
        self.param = {}

    def defend_real(self, trace):
        org_trace = trace.copy()
        org_trace[:, 0] = org_trace[:, 0] * 1000
        org_trace = org_trace.astype(int)
        new_column = np.full(len(org_trace), 536)
        trace = np.append(org_trace, new_column[:, np.newaxis], axis=1)
        trace = trace.astype(int)
        defended_trace = wtfpad.Wtfpad().run(trace)
        defended_trace = np.array(defended_trace).astype(float)
        defended_trace[:, 0] = defended_trace[:, 0] / 1000
        return defended_trace[:, :2]
