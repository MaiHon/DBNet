from .base import label2cls, cls2label
from .base import TextFuseBaseDS
from .synthtext import TextFuseNetSynthTextDS
from .icdar13 import TextFuseNetICDAR13DS


class CollateFN():
    def __init__(self):
        pass

    def __call__(self, batch):
        return tuple(zip(*batch))