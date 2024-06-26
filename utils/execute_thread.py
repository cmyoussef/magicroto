import json
import os
import platform
import signal
import subprocess
import threading
from typing import Callable, Dict, Union, Optional

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
    def __init__(self, args: Dict, callback: Optional[Callable], pre_cmd: str = None, post_cmd: str = None,
                 open_new_terminal: Union[bool, str] = False):
        """
        Initialize the thread with arguments, callback, optional pre and post commands, and a flag to open in a new terminal.

        @param args: A dictionary containing script arguments.
        @param callback: A callable function for the callback.
        @param pre_cmd: Optional pre-execution command.
        @param post_cmd: Optional post-execution command.
        @param open_new_terminal: Flag to open script in a new terminal.
        @return: None
        """
        threading.Thread.__init__(self)
        self.args = args
        self.callback = callback
        self.process = None
        self.terminal = None
        self.pre_cmd = pre_cmd or 'echo Starting'
        self.post_cmd = post_cmd or 'echo Execution finished'
        self.open_new_terminal = open_new_terminal
        if platform.system() != "Windows":
            self.open_new_terminal = False
        self._cmd = None
        self._pidfile = None

    @property
    def pidfile(self) -> str:
        """
        Generate and return the PID file path based on the user directory and terminal name.
        """
        if self._pidfile is None:
            user_dir = os.path.expanduser('~')
            filename = f"{self.open_new_terminal}_pidfile.txt" if isinstance(self.open_new_terminal,
                                                                             str) else "pidfile.txt"
            self._pidfile = os.path.join(user_dir, filename)
        return self._pidfile

    @property
    def cmd(self) -> Union[str, list]:
        """
        Generate and return the command string or list based on the arguments and conditions.
        """
        if self._cmd:
            return self._cmd
        # Extract necessary arguments
        export_env = self.args.pop('export_env', {})

        args = self.args.copy()
        python_exe = args.pop('python_exe')
        script_path = args.pop('script_path')
        
        # Check if the python interpreter exists and is a file
        if not os.path.isfile(python_exe):
            raise FileNotFoundError(f"The python interpreter '{python_exe}' does not exist or is not a file.")

        # Check if the python interpreter is executable
        if not os.access(python_exe, os.X_OK):
            raise PermissionError(f"The python interpreter '{python_exe}' is not executable.")

        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"The file '{script_path}' does not exist.")

        cmd = [python_exe, script_path]
        # setting cache dir
        cache_dir_str = ''
        if export_env:
            for k, v in export_env.items():
                if platform.system() == "Windows":
                    cache_dir_str += f'set {k}={v}\n'
                else:
                    cache_dir_str += f'export {k}={v}\n'
        if cache_dir_str:
            cmd.insert(0, cache_dir_str)

        # Process other arguments
        for key, value in args.items():
            cmd.append(f'--{key}')
            # Serialize list or dict to JSON string
            str_value = json.dumps(value)
            if isinstance(value, (list, )):
                cmd.append(f'"{str(str_value)}"')
            elif isinstance(value, (dict,)):
                cmd.append(json.dumps(str_value))
            else:
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

    @property
    def terminal_title(self):
        port = self.args.get('port', '')
        title = f"server port {port}" if port else "test_uniq title"
        if isinstance(self.open_new_terminal, str):
            title += self.open_new_terminal
        return title.replace(' ', '_')

    def run(self):
        """
        Run the Python script using subprocess, with optional pre and post commands, and in a new terminal if specified.
        """
        if self.open_new_terminal:

            if platform.system() == "Windows":
                cmd_lines = self.cmd.replace('\n', '& ')
                full_cmd = f'start /WAIT cmd.exe /k \"title {self.terminal_title} & {cmd_lines} & exit\"'
            else:
                cmd_lines = self.cmd.replace('\n', '; ')
                full_cmd = f'mate-terminal --title \"{self.terminal_title}\" -e \"bash -c \\\"{cmd_lines}; exec bash\\\"\"'
            self.terminal = subprocess.Popen(full_cmd, shell=True)
        else:
            # Original logic for running in the same terminal
            if platform.system() == "Windows":
                self.process = subprocess.Popen(["cmd.exe"], stdin=subprocess.PIPE)
            else:
                self.process = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE, preexec_fn=os.setsid)

            self.process.communicate(self.cmd.encode())  # Waits for the process to complete

            if use_call_back and self.callback is not None:
                nuke.executeInMainThread(self.callback)

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

        if self.terminal:
            # Terminate the new terminal window and its subprocesses
            if platform.system() == "Windows":
                # Terminate CMD window
                try:
                    terminal_process = psutil.Process(self.terminal.pid)
                    # Iterate over child processes and terminate them
                    for child in terminal_process.children(recursive=True):
                        child.terminate()
                    # After terminating children, terminate the CMD process itself
                    terminal_process.terminate()
                except psutil.NoSuchProcess:
                    pass  # Process might have already been terminated

            else:
                # For Unix-like systems, send SIGTERM to the process group
                os.killpg(os.getpgid(self.terminal.pid), signal.SIGTERM)
