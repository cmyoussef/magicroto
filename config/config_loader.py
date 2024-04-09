import inspect
import os

from nukebridge.config.config_loader import ConfigLoader as Base_ConfigLoader


class ConfigLoader(Base_ConfigLoader):

    def __init__(self):
        super().__init__()
        self.script_paths['RVM'] = os.path.join(self.project_directory, 'executors', 'rvmexecutor.py')

    @property
    def current_directory(self):
        """Get the current directory path."""
        return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
