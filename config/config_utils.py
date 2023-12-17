import inspect
import os
import json

import platform

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
DIR_PATH = os.path.dirname(current_dir)
python_exe = os.path.join(DIR_PATH, 'venv', 'Scripts', 'python')
python_exe = python_exe + '.exe' if platform.system() == "Windows" else python_exe
python_exe = python_exe if os.path.exists(python_exe) else ''

home_directory = os.path.expanduser("~")
tool_config_path = os.path.join(current_dir, "settings_config.json")
file_path = os.path.join(home_directory, ".magicRoto_config")

config_dict = {}
if os.path.exists(tool_config_path):
    with open(tool_config_path, 'r') as f:
        config_dict = json.load(f)
        config_dict['output_dir'] = os.path.join(home_directory, "magicRoto")
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        config_dict.update(json.load(f))

config_dict['python_path'] = config_dict.get('python_path', python_exe)

easy_roto_path = os.path.join(DIR_PATH, 'executors', 'easyrotoexecutor.py')
easy_roto_gui_path = os.path.join(DIR_PATH, 'executors', 'widgets', 'pointer.py')
mg_selector_live_path = os.path.join(DIR_PATH, 'executors', 'mrselectorlive.py')
mg_inpaint_path = os.path.join(DIR_PATH, 'executors', 'inpaintexecutor.py')
