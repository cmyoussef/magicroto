import inspect
import json
import os
import threading
import time

import nuke

from magicroto.config.config_utils import config_dict
from magicroto.gizmos.core.ui_doc import UIElementDocs
from magicroto.utils import common_utils
from magicroto.utils.execute_thread import ExecuteThread
from magicroto.utils.external_execute import run_external
from magicroto.utils.icons import Icons
from magicroto.utils.logger import logger_level, logger


class GizmoBase:
    _instances = {}  # Keep track of instances here
    unsupported_args = []
    MENU_GRP = 'generate'
    default_knobs = {}
    _initialized = False

    def __new__(cls, gizmo=None, name=None):

        logger.debug('run __new__')
        cls._instances.setdefault(cls.__name__, {})
        # gizmo = gizmo or name
        if gizmo and gizmo in cls._instances[cls.__name__]:
            instance = cls._instances[cls.__name__][gizmo]
            instance._initialized = True
            # instance._settings_is_loaded = True
            logger.debug('\t use old instance')
        else:
            instance = super().__new__(cls)
            instance._initialized = False
            logger.debug('\t create new instance')
            # instance._settings_is_loaded = False
        instance._settings_is_loaded = gizmo is not None
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
            self.gizmo.setName(name or f'ET_{self.__class__.__name__}')
            self.default_knobs = list(set(self.gizmo.knobs().keys()))

        # prevent initializing classes
        if self._initialized:
            logger.debug('\t stop init')
            return
        self.thread_list = []
        self.base_class = self.__module__.rsplit('.')[0]

        # store the instance to re-use it later
        self._instances[self.__class__.__name__][self.gizmo.name()] = self
        self.cache_dir = config_dict.get('cache_dir', '')
        self._python_path = config_dict.get('python_path', '')
        self.args = {'python_exe': self.python_path, 'unsupported_args': self.unsupported_args,
                     'cache_dir': self.cache_dir}
        self.gizmo.begin()
        self.output_node.setInput(0, self.input_node)
        if not self.gizmo.knob("gizmo_class_type"):
            self.gizmo_class_type = nuke.String_Knob("gizmo_class_type", "Gizmo class type")
            self.gizmo_class_type.setValue(self.__class__.__name__)
            self.gizmo.addKnob(self.gizmo_class_type)
            # self.gizmo_class_type.setVisible(False)

        if not self.gizmo.knob("gizmo_data_knob"):
            self.gizmo_data = nuke.Multiline_Eval_String_Knob("gizmo_data_knob", "Gizmo Data")
            self.gizmo.addKnob(self.gizmo_data)
            # self.gizmo_data.setVisible(False)

        if not self.gizmo.knob("default_knobs_knob"):
            self.default_knobs_knob = nuke.String_Knob("default_knobs_knob", "default knobs ")
            self.default_knobs_knob.setValue(json.dumps(self.default_knobs))
            self.gizmo.addKnob(self.default_knobs_knob)
            self.default_knobs_knob.setVisible(False)

        default_knobs = self.gizmo.knob("default_knobs_knob").value()
        self.default_knobs = json.loads(default_knobs) if default_knobs else []

        self.gizmo.knob('User').setFlag(nuke.INVISIBLE)
        self.gizmo.end()
        self.active_read_nodes = []
        self.user_tabs = []

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
        file_path = os.path.join(home_directory, ".easyTrack")
        try:
            with open(file_path, 'w') as f:
                json.dump(info_dict, f, indent=4)
            nuke.message(f"File successfully saved at {file_path}.")
        except Exception as e:
            nuke.message(f"An error occurred: {e}")

    def get_config_settings(self):
        return {
            'python_path': self.gizmo.knob("python_path_knob").value(),
            'cache_dir': self.gizmo.knob("cache_dir").value(),
            'output_dir': self.gizmo.knob("output_dir").value(),
            'pre_cmd': self.gizmo.knob("pre_cmd_knob").value(),
            'post_cmd': self.gizmo.knob("post_cmd_knob").value()
        }

    def save_config(self):

        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, ".easyTrack_config")
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
        if self._settings_is_loaded:
            return

        node = node or self.gizmo
        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, ".easyTrack")
        if os.path.exists(file_path):
            nuke.tprint(f'Loading settings from {file_path} on {node.name()}')
            with open(file_path, 'r') as f:
                info_dict = json.load(f)
            self.set_knob_info(info_dict, node)
        else:
            nuke.tprint(f"{file_path} file does not exist.")
        self.load_config()

    def load_config(self):
        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, ".easyTrack_config")
        fresh_load_config = config_dict
        if os.path.exists(file_path):
            print(f'loading from {file_path}')
            with open(file_path, 'r') as f:
                fresh_load_config = json.load(f)

        self.gizmo.knob("python_path_knob").setValue(fresh_load_config.get('python_path', ''))
        self.gizmo.knob('cache_dir').setValue(fresh_load_config.get('cache_dir', ''))
        self.gizmo.knob('output_dir').setValue(fresh_load_config.get('output_dir', ''))
        self.gizmo.knob('pre_cmd_knob').setValue(fresh_load_config.get('pre_cmd', ''))
        self.gizmo.knob('post_cmd_knob').setValue(fresh_load_config.get('post_cmd', ''))
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
            pre_cmd_knob.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(pre_cmd_knob)

        if not self.gizmo.knob('post_cmd_knob'):
            post_cmd_knob = nuke.Multiline_Eval_String_Knob('post_cmd_knob', 'post cmd')
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

    def create_generate_tab(self):
        if not self.gizmo.knob('generate'):
            generate_tab_knob = nuke.Tab_Knob("generate", 'Generate')
            self.gizmo.addKnob(generate_tab_knob)
        self.user_tabs.append('Generate')
        return self.gizmo.knob('generate')

    def create_generate_knobs(self):
        self.create_generate_tab()
        self.add_divider()

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

        if not self.gizmo.knob('interrupt_btn'):
            interrupt_btn = nuke.PyScript_Knob('interrupt_btn', f'Interrupt {Icons.interrupt_symbol}')
            self.gizmo.addKnob(interrupt_btn)
        self.gizmo.knob('interrupt_btn').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_interrupt")')

        self.set_status(running=False)

    @property
    def status_bar(self):
        status_bar = self.gizmo.knob('status_bar_knob')
        if not status_bar:
            status_bar = nuke.Text_Knob('status_bar_knob', f'Status Bar: ')
            self.gizmo.addKnob(status_bar)
        return status_bar

    def set_status(self, running=False, msg=''):
        executeBtn = self.gizmo.knob('execute_btn')

        if not executeBtn:
            return
        execute_btn = self.gizmo.knob('execute_btn')
        if running:
            if execute_btn:
                self.gizmo.knob('execute_btn').setFlag(nuke.DISABLED)
            msg = f'Running {msg} > check your terminal'
        else:
            if execute_btn:
                self.gizmo.knob('execute_btn').clearFlag(nuke.DISABLED)
            msg = f'Idle {msg}'
        self.status_bar.setValue(msg)

    @property
    def output_dir(self):
        return self.gizmo.knob("output_dir").value()

    @property
    def python_path(self):
        cache_dir_knob = self.gizmo.knob("python_path_knob")
        if cache_dir_knob:
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

    def knobChanged(self, knob):
        if knob.name() == 'controlNet_menu':
            pass

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
        return NotImplementedError()

    def get_output_dir(self):
        output_dir = self.gizmo.knob("output_dir").value()
        return os.path.join(output_dir, self.gizmo.name())

    def get_node(self, nodeName, nodeType=None):
        for node in self.gizmo.nodes():
            if node.name() == nodeName:
                return node

        if nodeType is None:
            print("Node type is none and the node doesn't exists")
            return

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

        for i in range(self.gizmo.maximumInputs()):
            if self.gizmo.input(i) == node:
                # Check if this input of the group node has a source connected
                return self.gizmo.input(i) is not None

    def writeInput(self, outputPath, node=None, ext='png'):
        node = node or self.input_node
        if not self.is_connected(node):
            self.gizmo.begin()
        # Use a Write node to save the input image
        write_node = nuke.nodes.Write()

        write_node.knob('file').setValue(outputPath.replace('\\', '/'))
        write_node.knob('channels').setValue('rgba')
        write_node.setInput(0, node)

        # Set the file_type and datatype
        write_node.knob('file_type').setValue(ext)
        write_node.knob('datatype').setValue('8 bit')

        # Execute the Write node
        nuke.execute(write_node.name(), nuke.frame(), nuke.frame())
        nuke.delete(write_node)
        self.gizmo.end()

    def get_init_img_path(self, img_name='init_img'):
        init_img_dir = os.path.join(self.get_output_dir(), f'source_{self.__class__.__name__}')
        os.makedirs(init_img_dir, exist_ok=True)

        init_img_path = os.path.join(init_img_dir, img_name+'.png')
        return init_img_path

    def on_execute(self):

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
        # Create a thread for each file
        timeout = self.gizmo.knob('time_out').value()
        files = zip(file_paths, node_names)
        thread = threading.Thread(target=self.check_files, args=(files, timeout))
        thread.start()
        self.thread_list.append(thread)
        return thread

    def on_interrupt(self):
        p = False
        for t in self.thread_list:
            try:
                t.terminate()
                p = True
            except:
                pass
        self.thread_list = []
        if p:
            nuke.tprint(f'Terminating running processes')
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
            logger.debug(f"running '{cls.__name__}' '{func_name}, Node Name: {nuke.thisNode().name()}'")
            return func(*args[1:], **kwargs)
        else:
            raise AttributeError(f"'{cls.__name__}' object has no method '{func_name}'")

    def update_single_read_node(self, node, file_path):
        file_path = file_path.replace('"', '').replace('\\', '/')
        node.knob('file').setValue(file_path)
        node.knob('reload').execute()
        self.force_evaluate_nodes()

    def check_files(self, files, timeout):
        time.sleep(5)
        start_time = time.time()
        remaining_files = set(files)  # Convert list to set for efficient removal.
        while remaining_files:
            for file_path, node in list(remaining_files):  # Create a copy of the set for iteration.
                if common_utils.check_file_complete(file_path):
                    # time.sleep(1)
                    nuke.executeInMainThread(self.update_single_read_node, args=(node, file_path,))
                    remaining_files.remove((file_path, node))  # Remove from the set.
            # Check if the timeout has been reached
            if time.time() - start_time > timeout:
                print("Timeout reached")
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

    def show_hide_knobs(self, knob_list, show=False):
        if isinstance(knob_list, str):
            knob_list = [knob_list]
        for knob in knob_list:
            knobC = self.gizmo.knob(knob)
            if knobC:
                knobC.setVisible(show)


if __name__ == '__main__':
    GizmoBase(name='TestBaseNode')
