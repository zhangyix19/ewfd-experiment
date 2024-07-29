import os
import sys

rootpath = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

from base import EWFDDefense, Empty
from front import Front
from tamaraw import Tamaraw
from wfgan import Wfgan
from regulator import RegulaTor
from frontamaraw import FronTamaraw
from adptamaraw import AdpTamaraw
# from switch import Switch
from wtfpad import Spring, Interspace, Wtfpad
from ezdef import Ezpadding, Ezfixed, Ezlinear, Ezfixedrate
from customized import Customized
from pred import Pred
