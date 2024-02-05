from .df import DF
from .rf import RF
from .base import DNNAttack


def get_attack(name):
    return globals()[name]
