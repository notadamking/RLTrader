from enum import Enum


class ProviderDateFormat(Enum):
    TIMESTAMP_UTC = 1
    TIMESTAMP_MS = 2
    DATE = 3
    DATETIME_HOUR_12 = 4
    DATETIME_HOUR_24 = 5
    DATETIME_MINUTE_12 = 6
    DATETIME_MINUTE_24 = 7
    CUSTOM_DATIME = 8
