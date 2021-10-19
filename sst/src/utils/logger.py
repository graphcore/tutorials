import logging


def get_logger() -> logging.Logger:
    logger = logging.getLogger('sst')
    logger.addHandler(logging.StreamHandler())
    return logger
