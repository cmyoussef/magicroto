import inspect
import json
import os
import socket
import threading
import time
import traceback
from datetime import datetime
import platform

import nuke

import magicroto
from magicroto.config.config_utils import config_dict
from magicroto.gizmos.core.ui_doc import UIElementDocs
from magicroto.utils import common_utils
from magicroto.utils.execute_thread import ExecuteThread
from magicroto.utils.external_execute import run_external
from magicroto.utils.icons import Icons
from magicroto.utils.logger import logger_level, logger
from magicroto.utils.soketserver import SocketServer


class GizmoBase:
    _instances = {}  # Keep track of instances here
    unsupported_args = []
    MENU_GRP = 'generate'
    default_knobs = {}
    _initialized = False
    frame_padding = '%04d'

    def __new__(cls, gizmo=None, name=None):

        cls._instances.setdefault(cls.__name__, {})
        if gizmo and gizmo in cls._instances[cls.__name__]:
            instance = cls._instances[cls.__name__][gizmo]
            instance._initialized = True
            
        else:
            instance = super().__new__(cls)
            instance._initialized = False
        return instance

    def __init__(self, gizmo=None, name=None):

        logger.debug('run __init__')
        if gizmo:
            self.gizmo = nuke.toNode(gizmo)
            if self.gizmo is None:
                raise f"{gizmo} dose not exists"
            self.default_knobs = []
        else:
            self.gizmo = nuke.createNode('Group', inpanel=False)
            self.gizmo.setName(name or f'MR_{self.__class__.__name__}')
            self.default_knobs = list(set(self.gizmo.knobs().keys()))

        # prevent initializing classes
        if self._initialized:
            logger.debug('\t stop init')
            return
        self._settings_is_loaded = False
        self.is_client_connected = None
        self.mask_client = None
        self.thread_list = []
        self.base_class = self.__module__.rsplit('.')[0]

        # store the instance to re-use it later
        self._instances[self.__class__.__name__][self.gizmo.name()] = self
        self.cache_dir = config_dict.get('cache_dir', '')
        self._python_path = config_dict.get('python_exe', '')
        self.args = {'python_exe': self.python_path, 'unsupported_args': self.unsupported_args,
                     'cache_dir': self.cache_dir}
        self.gizmo.begin()
        self.output_node.setInput(0, self.input_node)
        if not self.gizmo.knob("gizmo_class_type"):
            self.gizmo_class_type = nuke.String_Knob("gizmo_class_type", "Gizmo class type")
            self.gizmo_class_type.setValue(self.__class__.__name__)
            self.gizmo.addKnob(self.gizmo_class_type)
            self.gizmo_class_type.setVisible(False)

        if not self.gizmo.knob("gizmo_data_knob"):
            self.gizmo_data = nuke.Multiline_Eval_String_Knob("gizmo_data_knob", "Gizmo Data")
            self.gizmo.addKnob(self.gizmo_data)
            self.gizmo_data.setVisible(False)

        if not self.gizmo.knob("default_knobs_knob"):
            self.default_knobs_knob = nuke.String_Knob("default_knobs_knob", "default knobs ")
            self.default_knobs_knob.setValue(json.dumps(self.default_knobs))
            self.gizmo.addKnob(self.default_knobs_knob)
            self.default_knobs_knob.setVisible(False)

        default_knobs = self.gizmo.knob("default_knobs_knob").value()
        self.default_knobs = json.loads(default_knobs) if default_knobs else []

        # knob change connection
        cmd = f'{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("knobChanged")'
        self.gizmo.knob('knobChanged').setValue(cmd)

        self.gizmo.knob('User').setFlag(nuke.INVISIBLE)
        self.gizmo.end()
        self.active_read_nodes = []
        self.user_tabs = []
        nuke.addOnDestroy(self.on_destroy)

    # region client
    def ensure_server_connection(self):
        # self.write_input()
        if self.is_server_running():
            logger.info("Server is already running.")
        else:
            logger.info("Starting the server.")
            self.on_interrupt()
            self.start_server()

        if not self.is_client_connected:
            logger.info("Attempting to connect to the server.")
            self.attempt_reconnect()

    def is_server_running(self):
        """Check if the server is running by attempting to connect to the server's port."""
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            test_socket.connect(("localhost", self.main_port))
            test_socket.close()
            return True
        except socket.error:
            return False

    def connect_to_server(self):
        try:
            if self.mask_client is not None:
                self.mask_client.close()

            self.mask_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.mask_client.connect(("localhost", self.main_port))
            self.mask_client.setblocking(False)
            self.is_client_connected = True
            logger.info(f"Successfully connected to server at port {self.main_port}")
            # self.set_status(True, f"Successfully connected to server at port {self.main_port}", )
            nuke.executeInMainThread(self.set_status,
                                     args=(True, f"Successfully connected to server at port {self.main_port}",))
            # self.set_status(True, f"Successfully connected to server at port {self.main_port}")
            return True

        except ConnectionRefusedError:
            logger.warning(f"Connection to server at port {self.main_port} refused.")
            if self.mask_client is not None:
                self.mask_client.close()
            self.is_client_connected = False
            return False

        except Exception as e:
            logger.error(f"Error while connecting: {e}")
            if self.mask_client is not None:
                self.mask_client.close()
            self.is_client_connected = False
            return False

    def start_server(self):
        self.set_status(True, f"Starting server at port {self.main_port} << Check terminal")

        self.update_args()

        pre_cmd = self.gizmo.knob('pre_cmd_knob').value() or None
        post_cmd = self.gizmo.knob('post_cmd_knob').value() or None
        open_new_terminal = self.gizmo.knob('open_new_terminal').value() or None

        thread = ExecuteThread(self.args, None, pre_cmd, post_cmd, open_new_terminal=open_new_terminal)
        thread.start()
        self.thread_list.append(thread)

        # logger.info(f"Started server at port {self.main_port}")
        self.set_status(True, f"Starting server at port {self.main_port}")

    def attempt_reconnect(self):
        thread = threading.Thread(target=self._attempt_reconnect, args=())
        thread.start()
        logger.debug('Attempt reconnect on another thread')
        self.thread_list.append(thread)

    def _attempt_reconnect(self):
        retry_count = 0
        while not self.connect_to_server() and retry_count < 5:
            time.sleep(retry_count+1)  # Waiting for 1 second before retrying
            retry_count += 1
            logger.info(f"Retrying connection to server (Attempt {retry_count})")
        if retry_count == 5:
            logger.error("Failed to connect to the server after multiple attempts.")

    @property
    def main_port(self):
        port_knob = self.gizmo.knob('port_knob')
        if port_knob:
            return int(port_knob.value())
        else:
            return SocketServer.find_available_port()

    @main_port.setter
    def main_port(self, port):
        port_knob = self.gizmo.knob('port_knob')
        if port_knob:
            set_port_value = lambda: port_knob.setValue(int(port))
            nuke.executeInMainThread(set_port_value)

    def find_available_port_knob(self):
        port_knob = self.gizmo.knob('port_knob')
        if port_knob:
            self.main_port = SocketServer.find_available_port()
            # port_knob.setValue()

    @property
    def ports(self):
        return [self.main_port]

    def close_server(self):

        logger.debug(f'attempt to close server at port {self.main_port}, from {self.mask_client}')
        if self.mask_client:
            header = b'command::'
            try:
                if self.mask_client.fileno() != -1:
                    try:
                        self.mask_client.sendall(header + b'quit')
                        self.mask_client.sendall(header + b'quit')
                        self.mask_client.close()
                    except OSError as e:
                        # Handle error or log it
                        print(f"Error sending data: {e}")
                self.mask_client = None
            except ConnectionResetError:
                pass
            try:
                self.on_interrupt()
            except:
                pass
            logger.info("Closing Command sent.")

    # endregion

    @property
    def data(self):
        d = self.gizmo.knob("gizmo_data_knob").value()
        return common_utils.str_to_dict(d)

    @data.setter
    def data(self, d):
        logger.debug(f'data.setter {d}')
        self.gizmo.knob("gizmo_data_knob").setValue(common_utils.dict_to_str(d))

    def populate_ui(self):
        self.gizmo.begin()
        self.create_inputs()
        self.create_generate_knobs()
        self.create_settings_knobs()
        self.gizmo.end()
        self.populate_doc()
        self.load_settings()

    def update_cache_dir(self):
        cache_dir_node = self.gizmo.knob("cache_dir")
        if cache_dir_node:
            user_cache_dir = cache_dir_node.value()
            if os.path.exists(user_cache_dir):
                self.cache_dir = user_cache_dir
                return self.cache_dir

    def populate_doc(self, node=None):
        node = node or self.gizmo
        all_knobs = node.knobs()

        for knob_name, knob_obj in all_knobs.items():
            if knob_name in self.default_knobs:
                continue
            knob_obj.setTooltip(UIElementDocs.get_doc(knob_name))

    def get_knob_info(self, node=None):

        node = node or self.gizmo
        all_knobs = node.knobs()
        info_dict = {}
        current_tab = "Settings"
        for knob_name, knob_obj in all_knobs.items():

            if not knob_obj.enabled():
                continue

            # if not knob_obj.getFlag(nuke.DISABLED):
            #     print('\tknob_obj.getFlag(nuke.DISABLED)')
            #     continue

            if knob_name in self.default_knobs:
                continue

            if knob_obj.Class() in ['PyScript_Knob']:
                continue

            if knob_obj.Class() == "Tab_Knob":
                current_tab = knob_obj.label()

            if current_tab not in self.user_tabs:
                continue

            if current_tab not in info_dict:
                info_dict[current_tab] = {}

            knob_value = knob_obj.value()

            if isinstance(knob_value, (str, int, float, bool, list, dict)):
                knob_value_type = type(knob_value).__name__
                info_dict[current_tab][knob_name] = {'value': knob_value, 'value_type': knob_value_type}
            else:
                print(f"Skipping {knob_name} as it's not JSON-serializable")

        return info_dict

    def set_knob_info(self, info_dict, node=None):

        node = node or self.gizmo
        for tab, knobs in info_dict.items():
            for knob, details in knobs.items():
                if knob in node.knobs():
                    value = details['value']
                    value_type = details['value_type']

                    # Casting value based on the type
                    if value_type == 'int':
                        value = int(value)
                    elif value_type == 'float':
                        value = float(value)
                    elif value_type == 'bool':
                        value = bool(value)
                    # Add more types if necessary

                    node[knob].setValue(value)
                else:
                    print(f"Knob {knob} does not exist in the node.")

    def save_settings(self):

        self.save_config()
        info_dict = self.get_knob_info()
        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, ".magicRoto")
        try:
            with open(file_path, 'w') as f:
                json.dump(info_dict, f, indent=4)
            nuke.message(f"File successfully saved at {file_path}.")
        except Exception as e:
            nuke.message(f"An error occurred: {e}")

    def get_config_settings(self):
        return {
            'python_exe': self.gizmo.knob("python_path_knob").value(),
            'cache_dir': self.gizmo.knob("cache_dir").value(),
            'output_dir': self.gizmo.knob("output_dir").value(),
            'pre_cmd': self.gizmo.knob("pre_cmd_knob").value(),
            'post_cmd': self.gizmo.knob("post_cmd_knob").value()
        }

    def save_config(self):

        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, ".magicRoto_config")
        info_dict = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                info_dict = json.load(f)

        info_dict.update(self.get_config_settings())
        try:
            with open(file_path, 'w') as f:
                json.dump(info_dict, f, indent=4)
            nuke.tprint(f"File successfully saved at {file_path}.")
        except Exception as e:
            nuke.tprint(f"An error occurred: {e}")

    def load_settings(self, node=None):
        
        if hasattr(self, '_settings_is_loaded') and self._settings_is_loaded:
            return
        
        self._settings_is_loaded = True

        node = node or self.gizmo
        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, ".magicRoto")
        if os.path.exists(file_path):
            nuke.tprint(f'Loading settings from {file_path} on {node.name()}')
            with open(file_path, 'r') as f:
                info_dict = json.load(f)
            self.set_knob_info(info_dict, node)
        else:
            nuke.tprint(f"{file_path} file does not exist.")
        # self.load_config()

    def load_config(self):
        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, ".magicRoto_config")
        fresh_load_config = {}
        if os.path.exists(file_path):
            print(f'loading from {file_path}')
            with open(file_path, 'r') as f:
                fresh_load_config = json.load(f)
        
        fresh_load_config.update(config_dict)
        print('*'*100)
        print(fresh_load_config)
        print('*'*100)
        print(config_dict)
        print('*'*100)
        knobs = [
            ['python_path_knob', 'python_exe'],
            ['cache_dir', 'cache_dir'],
            ['output_dir', 'output_dir'],
            ['pre_cmd_knob', 'pre_cmd'],
            ['post_cmd_knob', 'post_cmd']
        ]
        for (knob, key) in knobs:
            knob_obj = self.gizmo.knob(knob)
            knob_obj.setValue(fresh_load_config.get(key, ''))
        self.cache_dir = self.gizmo.knob('cache_dir').value()
        self.cache_dir = self.gizmo.knob('cache_dir').value()

        if os.path.exists(file_path):
            nuke.message(f'loaded from {file_path}')

    def create_settings_tab(self):
        if not self.gizmo.knob('settings'):
            settings = nuke.Tab_Knob("settings", 'Settings')
            self.gizmo.addKnob(settings)
        self.user_tabs.append('Settings')
        return self.gizmo.knob('settings')

    def add_divider(self, name=None, node=None):
        node = node or self.gizmo

        # Get the current line number from the call stack
        current_line_number = inspect.currentframe().f_back.f_lineno
        base_string = str(name) + str(self.__class__.__name__) + str(current_line_number)
        if node.knob(base_string):
            return

        total_length = 80

        if name:
            name = f' {name} '
            dash_count = (total_length - (len(name) + 2)) // 2
            divider = '-' * dash_count + name + '-' * dash_count
        else:
            divider = '-' * total_length

        text_knob = nuke.Text_Knob(base_string, '|')
        text_knob.setValue(divider + ' |')
        text_knob.setFlag(nuke.STARTLINE)
        text_knob.setFlag(nuke.DISABLED)
        node.addKnob(text_knob)

    def create_settings_knobs(self):
        self.create_settings_tab()

        if not self.gizmo.knob('save_settings_btn'):
            cn_button = nuke.PyScript_Knob('save_settings_btn', f'Save Settings {Icons.save_style_symbol}')
            cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('save_settings_btn').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("save_settings")')

        if not self.gizmo.knob('load_settings_btn'):
            cn_button = nuke.PyScript_Knob('load_settings_btn', f'Load Settings {Icons.load_symbol}')
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('load_settings_btn').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("load_config")')

        self.add_divider("Advanced CMD")
        if not self.gizmo.knob('pre_cmd_knob'):
            pre_cmd_knob = nuke.Multiline_Eval_String_Knob('pre_cmd_knob', 'pre cmd')
            pre_cmd_knob.setValue(config_dict.get('pre_cmd', ''))
            pre_cmd_knob.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(pre_cmd_knob)

        if not self.gizmo.knob('post_cmd_knob'):
            post_cmd_knob = nuke.Multiline_Eval_String_Knob('post_cmd_knob', 'post cmd')
            post_cmd_knob.setValue(config_dict.get('post_cmd', ''))
            post_cmd_knob.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(post_cmd_knob)

        self.add_divider()
        # logger level
        if not self.gizmo.knob('logger_level_menu'):
            logger_level_menu = nuke.Enumeration_Knob('logger_level_menu', 'logger level', list(logger_level.keys()))
            logger_level_menu.setFlag(nuke.STARTLINE)
            logger_level_menu.setValue('info')
            self.gizmo.addKnob(logger_level_menu)

        # time_out
        if not self.gizmo.knob('time_out'):
            time_out = nuke.Int_Knob('time_out', 'Time out', 1000)
            time_out.setValue(1000)
            self.gizmo.addKnob(time_out)

        self.add_divider('Directory location')

        if not self.gizmo.knob("output_dir"):
            self.directory_knob = nuke.File_Knob("output_dir", "Output Directory")
            output_dir = config_dict.get('output_dir', os.path.join(os.path.expanduser("~"), 'nuke-stable-diffusion'))
            self.directory_knob.setValue(output_dir)
            self.gizmo.addKnob(self.directory_knob)

        if not self.gizmo.knob("cache_dir"):
            self.cache_dir_knob = nuke.File_Knob("cache_dir", "Cache dir")
            self.gizmo.addKnob(self.cache_dir_knob)
            self.cache_dir_knob.setValue(self._cache_dir)

        if not self.gizmo.knob("python_path_knob"):
            self.python_path_knob = nuke.File_Knob("python_path_knob", "Python Path")
            self.gizmo.addKnob(self.python_path_knob)
            self.python_path_knob.setValue(self._python_path)

        open_new_terminal = self.gizmo.knob('open_new_terminal')
        if not open_new_terminal:
            open_new_terminal = nuke.Boolean_Knob('open_new_terminal', 'open new terminal')
            open_new_terminal.setFlag(nuke.STARTLINE)
            open_new_terminal.setValue(True)
            self.gizmo.addKnob(open_new_terminal)

    def create_generate_tab(self):
        if not self.gizmo.knob('generate'):
            generate_tab_knob = nuke.Tab_Knob("generate", 'Generate')
            self.gizmo.addKnob(generate_tab_knob)
        self.user_tabs.append('Generate')
        return self.gizmo.knob('generate')

    def reload_button(self):
        reload_btn = self.gizmo.knob('reload_btn_knob')
        if not reload_btn:
            reload_btn = nuke.PyScript_Knob('reload_btn_knob', f'Reload {Icons.refresh_symbol}')
            reload_btn.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(reload_btn)
        reload_btn.setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("reload_read_nodes")')
        return reload_btn

    def reload_read_nodes(self):
        for n in self.gizmo.nodes():
            # Check if the current node is a Read node
            if n.Class() == 'Read':
                # Reload the Read node
                n['reload'].execute()
                logger.warning(n.name())
        self.force_evaluate_nodes()

    def create_generate_knobs(self):
        self.create_generate_tab()
        self.add_divider("Server Utils")

        if not self.gizmo.knob('port_knob'):
            port_knob = nuke.Int_Knob('port_knob', f'Port {Icons.key_symbol}')
            port_knob.setFlag(nuke.STARTLINE)
            # port_knob.setValue(SocketServer.find_available_port())
            port_knob.setValue(38671)
            self.gizmo.addKnob(port_knob)

        if not self.gizmo.knob('find_available_port_knob'):
            cn_button = nuke.PyScript_Knob('find_available_port_knob', f'{Icons.reuse_symbol}')
            # cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('find_available_port_knob').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("find_available_port_knob")')

        if not self.gizmo.knob('start_server'):
            cn_button = nuke.PyScript_Knob('start_server', f'Start{Icons.launch_gui_symbol}')
            cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('start_server').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("ensure_server_connection")')

        if not self.gizmo.knob('connect_to_server'):
            cn_button = nuke.PyScript_Knob('connect_to_server', f'Connect{Icons.link_symbol}')
            # cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('connect_to_server').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("attempt_reconnect")')

        if not self.gizmo.knob('close_server'):
            cn_button = nuke.PyScript_Knob('close_server', f'Close{Icons.link_symbol}')
            # cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('close_server').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("close_server")')

        if not self.gizmo.knob('interrupt_btn'):
            interrupt_btn = nuke.PyScript_Knob('interrupt_btn', f'Force terminate All{Icons.explosion_symbol}')
            interrupt_btn.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(interrupt_btn)
        self.gizmo.knob('interrupt_btn').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_interrupt")')
        # self.add_divider()

        # self.set_status(running=False)
        # self.add_divider()

    def create_execute_buttons(self):
        if not self.gizmo.knob('use_external_execute'):
            use_external_execute = nuke.Boolean_Knob('use_external_execute', f'Use Farm {Icons.tractor_symbol}')
            use_external_execute.setFlag(nuke.STARTLINE)
            # use_external_execute.setFlag(nuke.DISABLED)
            self.gizmo.addKnob(use_external_execute)

        if not self.gizmo.knob('execute_btn'):
            cn_button = nuke.PyScript_Knob('execute_btn', f'Execute {Icons.execute_symbol}')
            cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('execute_btn').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_execute")')

    @property
    def status_bar(self):
        status_bar = self.gizmo.knob('status_bar_knob')
        if not status_bar:
            status_bar = nuke.Text_Knob('status_bar_knob', f'Status Bar: ')
            self.gizmo.addKnob(status_bar)
        return status_bar

    def set_status(self, running=False, msg=''):
        execute_btn = self.gizmo.knob('execute_btn')
        if running:
            if execute_btn:
                self.gizmo.knob('execute_btn').setFlag(nuke.DISABLED)
            msg = f'Running {msg} > check your terminal'
        else:
            if execute_btn:
                self.gizmo.knob('execute_btn').clearFlag(nuke.DISABLED)
            msg = f'Idle {msg}'
        nuke.executeInMainThread(self.status_bar.setValue, args=(msg,))
        # self.status_bar.setValue(msg)

    @property
    def output_dir(self):
        return self.gizmo.knob("output_dir").value()

    @property
    def python_path(self):
        python_path_knob = self.gizmo.knob("python_path_knob")
        if python_path_knob:
            self._python_path = self.gizmo.knob("python_path_knob").value()
        return self._python_path

    @property
    def cache_dir(self):
        cache_dir_knob = self.gizmo.knob("cache_dir")
        if cache_dir_knob:
            self._cache_dir = self.gizmo.knob("cache_dir").value()
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cDir):
        self._cache_dir = cDir
        # if hasattr(self, 'gizmo') and self.gizmo.knob("cache_dir"):
        #     self.gizmo.knob("cache_dir").setValue(cDir)

    @property
    def additional_network_dir(self):
        if self.gizmo.knob("additional_network_dir"):
            return self.gizmo.knob("additional_network_dir").value()
        else:
            return self.cache_dir

    def knobChanged(self, knob=None):

        knob = knob or nuke.thisKnob()

        if knob.name() == 'use_frame_range_knobs':
            isEnabled = knob.value()
            if isEnabled:
                self.gizmo.knob('last_frame_knob').clearFlag(nuke.DISABLED)
                self.gizmo.knob('first_frame_knob').clearFlag(nuke.DISABLED)
            else:
                self.gizmo.knob('last_frame_knob').setFlag(nuke.DISABLED)
                self.gizmo.knob('first_frame_knob').setFlag(nuke.DISABLED)

        if knob.name() == 'controlNet_menu':
            pass

        if knob.name() == 'logger_level_menu':
            logger.setLevel(logger_level.get(self.gizmo.knob('logger_level_menu').value(), 20))

    def find_node_in_group(self, node_name):
        for node in self.gizmo.nodes():
            if node.name() == node_name:
                return node
        return None

    def create_inputs(self):
        return self.input_node

    @property
    def input_node(self):
        return self.get_node("Input1", 'Input')

    @property
    def output_node(self):
        return self.get_node("Output1", 'Output')

    def update_args(self):

        # Define a list of environment keys that will be used to set up the environment for the external process
        env_keys = ['HF_HOME', 'PIP_CACHE_DIR', 'TRANSFORMERS_CACHE']

        # Create a dictionary where each key from env_keys is associated with the cache directory path
        # This will be used to set up the environment variables for the external process
        self.args['export_env'] = dict([(k, self.cache_dir) for k in env_keys])

        # Get the directory path of the magicroto module
        magicroto_dir = os.path.dirname(os.path.dirname(os.path.abspath(magicroto.__file__)))
        magicroto_dir = magicroto_dir.replace('\\', '/')  # Replace backslashes with forward slashes for compatibility
        PYTHONPATH_ = '%PYTHONPATH%;' if platform.system() == "Windows" else '$PYTHONPATH:'
        # Add the magicroto directory path to the PYTHONPATH environment variable
        # This ensures that the external process has access to the magicroto module
        self.args['export_env']['PYTHONPATH'] = f'{PYTHONPATH_}"{magicroto_dir}"'

        self.args['logger_level'] = logger_level.get(self.gizmo.knob('logger_level_menu').value(), 20)
        self.args['ports'] = self.ports
        self.args['python_exe'] = self.python_path
        self.args['cache_dir'] = self.cache_dir

    @property
    def output_file_path(self):
        file_name = 'mask.png'
        output_dir_path = os.path.join(self.get_output_dir(), f'{datetime.now().strftime("%Y%m%d")}')
        output_dir_path = output_dir_path.replace('\\', '/')
        os.makedirs(output_dir_path, exist_ok=True)
        return self.add_padding(os.path.join(output_dir_path, file_name))

    def get_output_dir(self):
        output_dir = self.gizmo.knob("output_dir").value()
        return os.path.join(output_dir, self.gizmo.name())

    def get_node(self, nodeName, nodeType=None):
        for node in self.gizmo.nodes():
            if node.name() == nodeName:
                return node

        if nodeType is None:
            print("Node type is None and the node doesn't exists")
            return

        self.gizmo.begin()
        if nodeType == 'Input':
            node = nuke.nodes.Input()

        elif nodeType == 'Output':
            node = nuke.nodes.Output()
        else:
            node = nuke.createNode(nodeType, inpanel=False)

        node.setName(nodeName)

        return node

    def is_connected(self, node=None):
        # Find which index the internal input is mapped to in the group node
        if isinstance(node, str):
            node = self.get_node(node)

        node = node or self.input_node
        inp_index = int(str(node).split(' ')[-1])
        return True if self.gizmo.input(inp_index) else False

    def add_padding(self, path, ext=None):
        base_path, current_ext = os.path.splitext(path)
        ext = (ext or current_ext).replace('.', '')
        base_path = base_path.replace('\\', '/')
        path = f"{base_path}.{ext}" if f'.{self.frame_padding}.' in path else f"{base_path}.%04d.{ext}"
        return path

    def create_frame_range_knobs(self):

        if not self.gizmo.knob('first_frame_knob'):
            first_frame_knob = nuke.Int_Knob('first_frame_knob', ' ')
            nuke.root().knob('first_frame').value()
            first_frame_knob.setFlag(nuke.STARTLINE)
            # first_frame_knob.setFlag(nuke.DISABLED)
            first_frame_knob.setValue(int(nuke.root().knob('first_frame').value()))
            self.gizmo.addKnob(first_frame_knob)

        if not self.gizmo.knob('last_frame_knob'):
            end_frame_knob = nuke.Int_Knob('last_frame_knob', ' ')
            end_frame_knob.clearFlag(nuke.STARTLINE)
            # end_frame_knob.setFlag(nuke.DISABLED)
            end_frame_knob.setValue(int(nuke.root().knob('last_frame').value()))
            self.gizmo.addKnob(end_frame_knob)

        if not self.gizmo.knob('frame_range_label_knob'):
            frame_range_label_knob = nuke.Text_Knob('frame_range_label_knob', 'Frame Range')
            frame_range_label_knob.clearFlag(nuke.STARTLINE)
            self.gizmo.addKnob(frame_range_label_knob)

    @property
    def frame_range(self):

        start_frame = self.gizmo.knob('first_frame_knob').value()
        end_frame = self.gizmo.knob('last_frame_knob').value()

        return int(start_frame), int(end_frame)

    def writeInput(self, outputPath, node=None, ext='png', add_padding=False, frame_range=None, temp=False):
        node = node or self.input_node
        if not self.is_connected(node):
            self.gizmo.begin()

        # Use a Write node to save the input image
        write_node = nuke.nodes.Write()
        write_node['create_directories'].setValue(True)
        nodes_to_delete = [write_node]
        if temp:
            base_path, current_ext = os.path.splitext(outputPath)
            base_dir = os.path.dirname(base_path)
            base_name = os.path.basename(base_path)
            outputPath = os.path.join(base_dir, 'temp', f'{base_name}.{ext}')
            reformat_node = nuke.nodes.Reformat()
            nodes_to_delete.append(reformat_node)
            reformat_node.setInput(0, node)
            reformat_node['type'].setValue('to box')
            reformat_node['box_fixed'].setValue(True)
            reformat_node['box_width'].setValue(1)
            reformat_node['box_height'].setValue(1)
            write_node.setInput(0, reformat_node)

        outputPath = self.add_padding(outputPath, ext)
        write_node.knob('file').setValue(outputPath)
        write_node.knob('channels').setValue('rgba')
        write_node.setInput(0, node)

        # Set the file_type and datatype
        write_node.knob('file_type').setValue(ext)
        # Execute the Write node
        if frame_range is None:
            start_frame = end_frame = nuke.frame()
        else:
            start_frame, end_frame = frame_range
        # write_node.knob('datatype').setValue('8 bit')
        logger.debug(f'Output Path{outputPath}, frames {start_frame, end_frame}')

        nuke.execute(write_node.name(), start_frame, end_frame)
        for n in nodes_to_delete:
            nuke.delete(n)
        self.gizmo.end()

        if not add_padding:
            outputPath = outputPath.replace(f'.{self.frame_padding}.', f".{nuke.frame():04d}.")
        return outputPath

    def get_init_img_path(self, img_name='init_img'):
        # logger.debug(self.gizmo.name())
        init_img_dir = os.path.join(self.get_output_dir(), f'source_{self.gizmo.name()}')
        os.makedirs(init_img_dir, exist_ok=True)

        init_img_path = os.path.join(init_img_dir, img_name + '.png')
        return init_img_path

    def on_execute(self):
        self.update_args()

        pre_cmd = self.gizmo.knob('pre_cmd_knob').value() or None
        post_cmd = self.gizmo.knob('post_cmd_knob').value() or None
        thread = ExecuteThread(self.args, self.update_output_callback, pre_cmd, post_cmd)
        # print the command to that run the stable diffusion.
        nuke.tprint(f"{'-' * 100}\nExecute {self.__class__.__name__}:\n{thread.cmd}\n{'-' * 100}")

        if self.gizmo.knob('use_external_execute').value():
            nuke.tprint(f"{'-' * 100}\nExternal Execute")
            output = run_external(thread.cmd)
            nuke.tprint(f'{output}')
        else:
            thread.start()
            self.thread_list.append(thread)
            self.set_status(running=True)

    def update_output_callback(self, output_batch):
        self.gizmo.begin()
        if not isinstance(output_batch, list):
            output_batch = [output_batch]

        output_nodes = []
        num = 1
        for b, output_files in enumerate(output_batch):
            if not isinstance(output_files, list):
                output_files = [output_files]
            for i, output_file in enumerate(output_files):
                fileNode = self.get_node(f"Read{num}", 'Read')
                self.update_single_read_node(fileNode, output_file)
                fileNode.knob('reload').execute()
                output_nodes.append(fileNode)
                num += 1
        self.gizmo.end()
        self.set_status(running=False)
        return output_nodes

    def check_multiple_files(self, node_names, file_paths):
        if not isinstance(node_names, list):
            node_names = [node_names]
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        # Create a thread for each file
        timeout = self.gizmo.knob('time_out').value()
        files = zip(file_paths, node_names)
        thread = threading.Thread(target=self.check_files, args=(files, timeout))
        thread.start()
        self.thread_list.append(thread)
        return thread

    def on_interrupt(self):
        p = False
        self.close_server()
        for t in self.thread_list:
            if hasattr(t, 'terminate'):
                try:
                    t.terminate()
                    p = True
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.debug(f"An error occurred in terminate: {e}\n{tb}")
        self.thread_list = []
        if p:
            logger.warning(f'Terminating running processes')
        self.set_status(running=False)

    @classmethod
    def run_instance_method(cls, *args, **kwargs):
        node_name = nuke.thisNode().name()
        cls._instances.setdefault(cls.__name__, {})
        if node_name in cls._instances[cls.__name__]:
            inst = cls._instances[cls.__name__][node_name]
        else:
            inst = cls(nuke.thisNode().name())
        func_name = args[0]

        # return
        func = getattr(inst, func_name, None)
        if func and callable(func):
            # logger.debug(f"running '{cls.__name__}' '{func_name}, Node Name: {nuke.thisNode().name()}'")
            return func(*args[1:], **kwargs)
        else:
            raise AttributeError(f"'{cls.__name__}' object has no method '{func_name}'")

    def update_single_read_node(self, node, file_path):
        file_path = file_path.replace('"', '').replace('\\', '/')
        node.knob('file').setValue(file_path)
        node.knob('reload').execute()

        first, last = self.frame_range
        node.knob('first').setValue(first)
        node.knob('last').setValue(last)

        # Set the 'on_error' parameter to 'nearest frame'
        node.knob('on_error').setValue('black')
        self.force_evaluate_nodes()

    def check_files(self, files, timeout, sleep=.1):
        time.sleep(5)
        start_time = time.time()
        # Convert list to set for efficient removal.
        remaining_files = set(files)
        while remaining_files:
            for file_path, node in list(remaining_files):  # Create a copy of the set for iteration.
                file_path_to_check = file_path.replace(f'.{self.frame_padding}.', f'.{self.frame_range[0]:04d}.')
                if common_utils.check_file_complete(file_path_to_check):
                    # time.sleep(sleep)
                    nuke.executeInMainThread(self.update_single_read_node, args=(node, file_path,))
                    logger.debug(f'updating {file_path}')
                    remaining_files.remove((file_path, node))  # Remove from the set.
            # Check if the timeout has been reached
            if time.time() - start_time > timeout:
                logger.warning("Timeout reached, It will stop look for the image to be rendered")
                break
            time.sleep(2)

    @staticmethod
    def disconnect_inputs(node):
        for i in range(node.inputs()):
            node.setInput(i, None)

    def force_evaluate_nodes(self):
        node = self.get_node('Output1', 'Output')

        # for n in self.gizmo.nodes():
        knob = node.knob('label')
        current_label = knob.value()
        knob.setValue(current_label + " ")
        knob.setValue(current_label)

    @staticmethod
    def _force_evaluate_node(node):
        # node.showControlPanel()
        node.hideControlPanel()
        node.showControlPanel()
        node.redraw()
        node.hideControlPanel()
        knob = node.knob('label')
        current_label = knob.value()
        knob.setValue(current_label + " ")
        knob.setValue(current_label)

    def show_hide_knobs(self, knob_list, show=False):
        if isinstance(knob_list, str):
            knob_list = [knob_list]
        for knob in knob_list:
            knobC = self.gizmo.knob(knob)
            if knobC:
                knobC.setVisible(show)

    def on_destroy(self):
        if nuke.thisNode() == self.gizmo:
            self.close_server()


if __name__ == '__main__':
    GizmoBase(name='TestBaseNode')
