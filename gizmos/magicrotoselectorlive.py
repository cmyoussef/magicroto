import glob
import os
import socket
import threading
import time
from datetime import datetime

import nuke
from PIL import Image
from magicroto.config.config_utils import mg_selector_live_path
from magicroto.gizmos.core.gizmobase import GizmoBase
from magicroto.utils import image_utils
from magicroto.utils.execute_thread import ExecuteThread
from magicroto.utils.icons import Icons
from magicroto.utils import common_utils
from magicroto.utils.logger import logger, logger_level
from magicroto.utils.soketserver import SocketServer


class MagicRotoSelectorLive(GizmoBase):

    def __init__(self, node=None, name=None):
        super().__init__(gizmo=node, name=name)

        self.knob_change_timer = None
        self.last_knob_value = None

        self.mask_port = SocketServer.find_available_port()

        self.mask_client = None
        # Variable to store the image
        if not self._initialized:
            self.is_client_connected = False  # Flag to track connection status

            self.args['script_path'] = mg_selector_live_path
            self.populate_ui()

        self.args = {'python_path': self.args.get('python_path'),
                     'cache_dir': self.args.get('cache_dir'),
                     'SAM_checkpoint': os.path.join(self.args.get('cache_dir'), 'sam_vit_h_4b8939.pth'),
                     'model_type': 'vit_h',
                     'device': "cuda:0",

                     'script_path': mg_selector_live_path}
        self.gizmo.begin()
        # Creating or getting the plus_mask node and setting its operation to "plus"
        self.plus_merge_node = self.get_node("plus_mask", 'Merge2')
        self.plus_merge_node['operation'].setValue("over")
        self.plus_merge_node['mix'].setValue(.5)
        # Clear selection
        nuke.selectAll()
        nuke.invertSelection()
        # Creating or getting the mask_combine node

        # 1. Connect the input_node to the B input of plus_mask
        self.plus_merge_node.setInput(0, self.input_node)

        # 2. Connect mask_combine to the A input of plus_mask
        self.plus_merge_node.setInput(1, self.get_node(f"Read1", 'Read'))

        # 3. Connect plus_merge_node to output_node
        self.output_node.setInput(0, self.plus_merge_node)

        # End of gizmo modification
        self.gizmo.end()

        self.points = []
        self.counter = 0
        nuke.addOnDestroy(self.on_destroy)

    def write_input(self):
        # if not self.is_connected(self.input_node):
        #     return

        init_img_path = self.get_init_img_path()
        init_img_path_padding = self.add_padding(init_img_path)
        existing_file = init_img_path_padding.replace(f'.{self.frame_padding}.', f".{nuke.frame():04d}.")

        self.writeInput(init_img_path, self.input_node)
        return existing_file

    def ensure_server_connection(self):
        # self.write_input()
        if self.is_server_running():
            logger.info("Server is already running.")
        else:
            logger.info("Starting the server.")
            self.start_server()

        if not self.is_client_connected:
            logger.info("Attempting to connect to the server.")
            self.attempt_reconnect()

    def is_server_running(self):
        """Check if the server is running by attempting to connect to the server's port."""
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            test_socket.connect(("localhost", self.mask_port))
            test_socket.close()
            return True
        except socket.error:
            return False

    def connect_to_server(self):
        try:
            if self.mask_client is not None:
                self.mask_client.close()

            self.mask_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.mask_client.connect(("localhost", self.mask_port))
            self.mask_client.setblocking(False)
            self.is_client_connected = True
            logger.info(f"Successfully connected to server at port {self.mask_port}")
            self.set_status(True, f"Successfully connected to server at port {self.mask_port}")
            return True
        except ConnectionRefusedError:
            logger.warning(f"Connection to server at port {self.mask_port} refused.")
            self.mask_client.close()
            self.is_client_connected = False
            return False
        except Exception as e:
            logger.error(f"Error while connecting: {e}")
            self.is_client_connected = False
            return False

    def start_server(self):
        self.set_status(True, f"Starting server at port {self.mask_port} << Check terminal")

        self.update_args()

        pre_cmd = self.gizmo.knob('pre_cmd_knob').value() or None
        post_cmd = self.gizmo.knob('post_cmd_knob').value() or None

        thread = ExecuteThread(self.args, None, pre_cmd, post_cmd)
        thread.start()
        self.thread_list.append(thread)

        logger.info(f"Started server at port {self.mask_port}")
        self.set_status(True, f"Started server at port {self.mask_port}")

    def attempt_reconnect(self):
        retry_count = 0
        while not self.connect_to_server() and retry_count < 5:
            time.sleep(1)  # Waiting for 1 second before retrying
            retry_count += 1
            logger.info(f"Retrying connection to server (Attempt {retry_count})")
            if retry_count == 4:
                self.mask_port = SocketServer.find_available_port()
        if retry_count == 5:
            logger.error("Failed to connect to the server after multiple attempts.")

    @property
    def input_node(self):
        return self.get_node("image", 'Input')

    @property
    def mask_input_node(self):
        return self.get_node("mask", 'Input')

    def create_generate_knobs(self):
        self.create_generate_tab()

        if not self.gizmo.knob('connect_to_server'):
            cn_button = nuke.PyScript_Knob('connect_to_server', f'Connect to server{Icons.launch_gui_symbol}')
            cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('connect_to_server').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("ensure_server_connection")')

        self.add_divider()

        # region frame rang
        if not self.gizmo.knob('first_frame_knob'):
            first_frame_knob = nuke.Int_Knob('first_frame_knob', ' ')
            nuke.root().knob('first_frame').value()
            first_frame_knob.setFlag(nuke.STARTLINE)
            first_frame_knob.setFlag(nuke.DISABLED)
            first_frame_knob.setValue(int(nuke.root().knob('first_frame').value()))
            self.gizmo.addKnob(first_frame_knob)

        if not self.gizmo.knob('last_frame_knob'):
            end_frame_knob = nuke.Int_Knob('last_frame_knob', ' ')
            end_frame_knob.clearFlag(nuke.STARTLINE)
            end_frame_knob.setFlag(nuke.DISABLED)
            end_frame_knob.setValue(int(nuke.root().knob('last_frame').value()))
            self.gizmo.addKnob(end_frame_knob)

        if not self.gizmo.knob('use_frame_range_knobs'):
            use_frame_range_knobs = nuke.Boolean_Knob('use_frame_range_knobs',
                                                      f'{Icons.video_symbol} Use frame range')
            use_frame_range_knobs.clearFlag(nuke.STARTLINE)
            self.gizmo.addKnob(use_frame_range_knobs)
        # endregion
        self.add_divider()

        if not self.gizmo.knob('track_01'):
            track_01_xy = nuke.XY_Knob('track_01', f'Track 01')
            track_01_xy.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(track_01_xy)

        self.add_divider('masks')

        for i in range(1, 3):
            knobLabel = '{:02d}'.format(i)
            knobName = f'mask_{knobLabel}_knob'
            if not self.gizmo.knob(knobName):
                msk_knob = nuke.Boolean_Knob(knobName, knobLabel)
                self.gizmo.addKnob(msk_knob)
                if i == 1 or i % 10 == 1:
                    msk_knob.setFlag(nuke.STARTLINE)
                msk_knob.setVisible(False)
                msk_knob.setValue(1)

        status_bar = self.status_bar
        self.set_status(running=False)
        # super().create_generate_knobs()

    def knobChanged(self, knob=None):

        knob = knob or nuke.thisKnob()

        if knob.name() in ['track_01', 'frame']:
            xy = int(knob.value()[0]), int(knob.value()[1])
            # Check if the value has changed
            if xy != self.last_knob_value:
                # Update the last known value
                self.last_knob_value = xy

                # Cancel the existing timer if it's running
                if self.knob_change_timer is not None:
                    self.knob_change_timer.cancel()

                # Create a new timer
                self.knob_change_timer = threading.Timer(0.5, self.on_points_changed, args=(xy,))
                self.knob_change_timer.start()

        super().knobChanged(knob)

    def on_points_changed(self, xy_points):

        init_img_path = self.get_init_img_path()
        init_img_path_padding = self.add_padding(init_img_path)
        existing_file = init_img_path_padding.replace(f'.{self.frame_padding}.', f".{nuke.frame():04d}.")
        if not os.path.exists(existing_file):
            nuke.executeInMainThread(self.writeInput, args=(init_img_path, self.input_node,))
            time.sleep(1)

        else:
            init_img_path = None
        safety_break = 0
        while not common_utils.check_file_complete(existing_file):
            time.sleep(1)
            safety_break += 1
            if safety_break > 10:
                logger.error(f'Not able to load {existing_file}')

        y_shift = Image.open(existing_file).height

        xy_points = int(xy_points[0]), y_shift - int(xy_points[1])

        data_to_send = {
            'use_xy': True,
            'frame': nuke.frame(),
            'init_img_path': init_img_path_padding,
            'prompts': {
                'point_coords': [xy_points],
                'point_labels': [1]
            }
        }
        while True:
            try:
                self.mask_client.sendall(SocketServer.encode_data(data_to_send.copy()))
                break

            except ConnectionResetError:
                self.ensure_server_connection()

            except AttributeError:
                self.ensure_server_connection()

        file_paths = self.output_file_path.replace(f'.{self.frame_padding}.', f'.{nuke.frame():04d}.')
        self.check_multiple_files([self.get_node(f"Read1", 'Read')], [file_paths])


    @property
    def output_file_path(self):
        output_dir_path = os.path.join(self.get_output_dir(), f'{datetime.now().strftime("%Y%m%d")}')
        output_dir_path = output_dir_path.replace('\\', '/')
        os.makedirs(output_dir_path, exist_ok=True)
        return self.add_padding(os.path.join(output_dir_path, 'mask.png'))

    def update_args(self):
        super().update_args()

        self.args['image'] = self.write_input()
        self.args['python_path'] = self.python_path
        self.args['cache_dir'] = self.cache_dir
        self.args['output_path'] = self.output_file_path
        self.args['logger_level'] = logger_level.get(self.gizmo.knob('logger_level_menu').value(), 20)
        self.args['ports'] = [self.mask_port]
        self.args['script_path'] = mg_selector_live_path
        self.args['cache_dir'] = self.cache_dir
        self.args['SAM_checkpoint'] = os.path.join(self.args.get('cache_dir'), 'sam_vit_h_4b8939.pth')
        self.args['python_path'] = self.python_path

    def update_single_read_node(self, node, file_path):
        file_path = file_path.replace('"', '').replace('\\', '/')
        file_main_path, fn, ext = file_path.rsplit('.', 2)
        all_files = glob.glob(f'{file_main_path}.*.{ext}')
        frames = [int(f.rsplit('.', 2)[-2]) for f in all_files]

        wildcard_path = f'{file_main_path}.####.{ext}'
        node.knob('file').setValue(wildcard_path)
        node.knob('reload').execute()
        node.knob('first').setValue(min(frames))
        node.knob('last').setValue(max(frames))

        # Set the 'on_error' parameter to 'nearest frame'
        node.knob('on_error').setValue('nearest')

        # set keyFrames
        keys = self.gizmo.knob('keys_knob')
        # Make sure the knob is set to be animated
        keys.setAnimated()
        for frame in frames:
            logger.debug(f'setting {frame}')
            keys.setValueAt(frame, frame)

        self.force_evaluate_nodes()

    def close_server(self):
        if self.mask_client:
            header = b'command::'
            self.mask_client.sendall(header + b'quit')
            self.mask_client.close()
            self.mask_client = None
            logger.info("Server closed.")

    def on_destroy(self):
        if nuke.thisNode() == self.gizmo:
            self.close_server()

if __name__ == '__main__':
    # Create a NoOp node on which we'll add the knobs
    node = MagicRotoSelectorLive()
    node.showControlPanel()
