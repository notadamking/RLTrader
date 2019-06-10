import os
import logging
import colorlog

def init_logger(dunder_name, testing_mode) -> logging.Logger:
    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(levelname)s - '
        '%(message)s'
    )
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq} '
        '%(log_color)s '
        f'{log_format}'
    )
    colorlog.basicConfig(format=colorlog_format)
    logger = logging.getLogger(dunder_name)

    if testing_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    ## Note: these file outputs are left in place as examples
    ## Feel free to uncomment and use the outputs as you like

    # Output full log
    # fh = logging.FileHandler(os.path.join('log', 'trading.log')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    # # Output warning log
    # fh = logging.FileHandler(os.path.join('log', 'trading.warning.log')
    # fh.setLevel(logging.WARNING)
    # formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    # # Output error log
    # fh = logging.FileHandler(os.path.join('log', 'trading.error.log')
    # fh.setLevel(logging.ERROR)
    # formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    return logger
