from enum import Enum

class Gesture(Enum):
    OTHER   = -1
    IDLE    = 0
    STOP    = 1
    FORWARD = 2
    REVERSE = 3
    HAIL    = 4
    LEFT    = 5
    RIGHT   = 6
    FOLLOW  = 7

    @classmethod
    def get(cls, value, default=None):
        return cls._value2member_map_.get(value, default).name