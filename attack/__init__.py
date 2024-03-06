from .df import DF
from .rf import RF
from .base import Attack, DNNAttack, MLAttack
from .kfp import KFP


def get_attack(name):
    return globals()[name]
