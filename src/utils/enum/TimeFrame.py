
from enum import Enum, unique

@unique
class TimeFrame(Enum):
    DAILY = 1
    WEEKLY = 2
    MONTHLY = 3