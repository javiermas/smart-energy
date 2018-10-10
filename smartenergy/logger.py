from os import getenv
from uuid import uuid4
import logging


class Logger(object):

    def __init__(self, class_name):
        self.class_name = class_name
        self._set_up_logging()

    def _set_up_logging(self):
        self.log = logging.getLogger(self.class_name + str(uuid4()))
        self.log.setLevel(getenv('{}_LOG_LEVEL'.format(self.class_name.upper()), 'INFO'))
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(levelname)s:::{self.class_name}:::%(asctime)s:::%(message)s')
        handler.setFormatter(formatter)
        self.log.addHandler(handler)

    def info(self, text):
        self.log.info(text)

    def debug(self, text):
        self.log.debug(text)
    
