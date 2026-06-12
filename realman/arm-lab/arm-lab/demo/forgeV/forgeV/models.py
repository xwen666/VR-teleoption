from dataclasses import dataclass
from collections.abc import Iterator
import math


@dataclass
class Pos:
    """
    位置: (x,y,z)     m
    姿态: (rx,ry,rz)  rad
    """

    x: float = 0
    y: float = 0
    z: float = 0
    rx: float = 0
    ry: float = 0
    rz: float = 0

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z
        yield self.rx
        yield self.ry
        yield self.rz

    def distance(self, pos: 'Pos') -> float:
        return math.hypot(pos.x - self.x, pos.y - self.y, pos.z - self.z)
