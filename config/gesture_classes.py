from enum import Enum

class Gesture(Enum):
    def __new__(cls, value, color_rgb, icon):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.color = color_rgb
        obj.icon = icon
        return obj

    IDLE    = 0, (255, 153,   0), "pause"
    TRANS   = 1, (150,   0, 150), "sync"
    STOP    = 2, (  0,   0, 255), "hand"
    FORWARD = 3, (  0, 255,   0), "arrow-up"
    REVERSE = 4, (  0, 255,   0), "arrow-down"
    HAIL    = 5, (  0,   0, 255), "wave"
    LEFT    = 6, (  0, 255,   0), "arrow-left"
    RIGHT   = 7, (  0, 255,   0), "arrow-right"
    FOLLOW  = 8, (153,   0, 255), "user-follow"
    POINT   = 9, ( 51,  51, 255), "hand-point"

    def __str__(self):
        return f"{self.label}"

    @classmethod
    def from_label(cls, label: str):
        for gesture in cls:
            if gesture.label.lower() == label.lower():
                return gesture
        return None