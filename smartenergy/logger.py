from os import getenv
from uuid import uuid4
import logging


class Logger(object):

    def __init__(self):
        self._set_up_logging()

    def _set_up_logging(self):
        class_name = self.__class__.__name__
        self.log = logging.getLogger(class_name + str(uuid4()))
        self.log.setLevel(getenv('{}_LOG_LEVEL'.format(class_name.upper()), 'INFO'))
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(levelname)s:::{class_name}:::%(asctime)s:::%(message)s')
        handler.setFormatter(formatter)
        self.log.addHandler(handler)

    def info(self, text):
        self.log.info(text)

    def debug(self, text):
        self.log.debug(text)
    
