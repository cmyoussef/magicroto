import logging
import threading


class ColorfulFormatter(logging.Formatter):
    format_dict = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",  # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold red
    }

    def format(self, record):
        color_format = self.format_dict.get(record.levelno)
        if color_format:
            record.levelname = f"{color_format}{record.levelname}\033[0m"
            record.msg = f"{color_format}{record.msg}\033[0m"
        return super().format(record)


class SingletonLogger:
    _instance_lock = threading.Lock()
    _instance = None

    @staticmethod
    def get_instance():
        if SingletonLogger._instance is None:
            with SingletonLogger._instance_lock:
                if SingletonLogger._instance is None:
                    SingletonLogger._instance = SingletonLogger()
        return SingletonLogger._instance

    def __init__(self, name='EasyRoto'):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        if not self.logger.handlers:  # This check prevents adding multiple handlers
            self.logger.setLevel('DEBUG')
            ch = logging.StreamHandler()
            ch.setLevel('DEBUG')
            formatter = ColorfulFormatter(
                "%(levelname)s [%(asctime)s - %(name)s:%(module)s:%(lineno)s - %(funcName)s()] || %(message)s",
                datefmt="%d-%b-%y %H:%M:%S")
            ch.setFormatter(formatter)
            for i in self.logger.handlers:
                if str(type(i)) == str(type(self.ch)):
                    self.logger.removeHandler(i)
            self.logger.addHandler(ch)


logger = SingletonLogger.get_instance().logger
logger_level = {
    'critical': 50,
    'error': 40,
    'warning': 30,
    'info': 20,
    'debug': 10,
    'notset': 0
}

logger.setLevel(20)
# Example usage
if __name__ == "__main__":
    logger.debug("This is a debug message.")
