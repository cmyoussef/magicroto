import json
import subprocess
import threading
from typing import Callable, Dict, Union, Optional
import os
import signal
import platform
import psutil

preexec = None
if platform.system() in ["Linux", "Darwin"]:
    preexec = os.setsid

use_call_back = False
try:
    import nuke
    use_call_back = True
except ModuleNotFoundError:
    pass


class ExecuteThread(threading.Thread):
    def __init__(self, args: Dict, callback: Optional[Callable], pre_cmd: str = None, post_cmd: str = None):
        """
        Initialize the thread with arguments, callback, and optional pre and post commands.

        :param args: A dictionary containing script arguments.
        :param callback: A callable function for the callback.
        :param pre_cmd: A string containing pre-execution commands.
        :param post_cmd: A string containing post-execution commands.
        """
        threading.Thread.__init__(self)
        self.args = args
        self.callback = callback
        self.process = None
        self.pre_cmd = pre_cmd or 'echo Starting'
        self.post_cmd = post_cmd or 'echo Excution finished'
        self._cmd = None

    @property
    def cmd(self) -> Union[str, list]:
        """
        Generate and return the command string or list based on the arguments and conditions.
        """
        if self._cmd:
            return self._cmd
        # Extract necessary arguments
        args = self.args.copy()
        python_exe = args.pop('python_exe')
        script_path = args.pop('script_path')

        cmd = [python_exe, script_path]
        # setting cache dir
        cache_dir = self.args.get('cache_dir', None)
        cache_dir_str = ''
        if cache_dir and not platform.system() == "Windows":
            for v in ['HF_HOME', 'PIP_CACHE_DIR', 'TRANSFORMERS_CACHE']:
                cache_dir_str += f'export {v}={cache_dir}\n'
            cmd.insert(0, cache_dir_str)
        else:
            cmd.insert(0, '')

        # Process other arguments
        for key, value in args.items():
            # Serialize list or dict to JSON string
            if isinstance(value, (list, )):
                value = json.dumps(value)
            cmd.append(f'--{key}')
            # quote everything if we are going to pass as string not a list
            cmd.append(f'"{str(value)}"')

        # If pre- or post-commands exist, run them along with the main command
        cmd = " ".join(cmd)
        if self.pre_cmd:
            cmd = f'{self.pre_cmd}\n{cmd}'
        if self.post_cmd:
            cmd = f'{cmd}\n{self.post_cmd}'
        self._cmd = cmd

        while self._cmd.startswith(' '):
            self._cmd = self._cmd[1:]

        return self._cmd

    def run(self):
        """
        Run the Python script using subprocess, with optional pre and post commands.
        """
        # If pre- or post-commands exist, run them with shell=True
        # self.process = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE) #  , shell=True
        if platform.system() == "Windows":
            self.process = subprocess.Popen(["cmd.exe"], stdin=subprocess.PIPE)
        else:
            self.process = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE, preexec_fn=os.setsid)

        self.process.communicate(self.cmd.encode())  # Waits for the process to complete

        # Execute callback in main Nuke thread
        if use_call_back and self.callback is not None:
            nuke.executeInMainThread(self.callback, args=(self.args.get('output', None),))

    def terminate(self):
        """
        Terminate the running process, if it exists.
        """
        if self.process:
            if platform.system() == "Windows":
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):  # or parent.children() for recursive=False
                    child.terminate()
                parent.terminate()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
