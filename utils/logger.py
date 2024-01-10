import logging
import threading

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


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


class SingletonLogger(logging.Logger):
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, name='MagicRoto'):
        if name not in cls._instances:
            with cls._lock:
                if name not in cls._instances:
                    logging.setLoggerClass(cls)  # Set the custom logger class
                    logger = logging.getLogger(name)
                    logger.setLevel('DEBUG')
                    ch = logging.StreamHandler()
                    ch.setLevel('DEBUG')
                    formatter = ColorfulFormatter(
                        "%(levelname)s [%(asctime)s - %(name)s:%(module)s:%(lineno)s - %(funcName)s()] || %(message)s",
                        datefmt="%d-%b-%y %H:%M:%S")
                    ch.setFormatter(formatter)
                    logger.addHandler(ch)
                    cls._instances[name] = logger
        return cls._instances[name]

    # Add the progress method directly to the logger
    def progress(self, iterable, desc="Processing", level=logging.INFO, **kwargs):
        if tqdm is None:
            for item in iterable:
                yield item
        # Get the color corresponding to the log level
        bar_color = ColorfulFormatter.format_dict.get(level, "\033[0m")  # Default to no color

        # Custom tqdm class to change color
        class ColoredTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def format_meter(self, *args, **kwargs):
                return bar_color + super().format_meter(*args, **kwargs) + '\033[0m'

        self.log(logging.DEBUG, f"{desc} started.")
        for item in ColoredTqdm(iterable, desc=desc, **kwargs):
            yield item
        self.log(logging.DEBUG, f"{desc} completed.")


logger = SingletonLogger.get_instance()
logger_level = {
    'critical': 50,
    'error': 40,
    'warning': 30,
    'info': 20,
    'debug': 10,
    'notset': 0
}

logger.setLevel(10)
# Example usage
if __name__ == "__main__":
    import time

    for _ in logger.progress(range(100), desc="Example Progress"):
        time.sleep(.1)
        # Your processing code here
        pass
