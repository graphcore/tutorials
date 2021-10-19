import logging


def disable_logging():
    logging.disable(logging.CRITICAL)


def enable_logging():
    logging.disable(logging.NOTSET)


class MockHandler(logging.Handler):
    def __init__(self, stream):
        super().__init__()
        self.log_records = stream

    def emit(self, record):
        self.log_records.append(record.getMessage())
