import glob
import os
import socket
import threading
import time
from datetime import datetime

import nuke

from magicroto.config.config_utils import mg_selector_live_path
from magicroto.gizmos.core.gizmobase import GizmoBase
from magicroto.utils import image_utils
from magicroto.utils.execute_thread import ExecuteThread
from magicroto.utils.icons import Icons
from magicroto.utils.logger import logger, logger_level
from magicroto.utils.soketserver import SocketServer


class MagicRotoSelectorLive(GizmoBase):

    def __init__(self, node=None, name=None):
        super().__init__(gizmo=node, name=name)

        self.knob_change_timer = None
        self.last_knob_value = None

        running_servers = SocketServer.get_instances()
        logger.warning(f'running_servers:{running_servers}')
        # if running_servers:
        #     self.mask_port = list(running_servers.keys())[0]
        # else:
        self.mask_port = 56400 # SocketServer.find_available_port()

        self.mask_client = None
        # Variable to store the image
        if not self._initialized:
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

    def connect_client(self):
        safety_break = 0
        while True:
            safety_break += 1

            # Close the existing socket if it's open
            if self.mask_client is not None:
                self.mask_client.close()

            # Create a new socket
            self.mask_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            try:
                self.mask_client.connect(("localhost", self.mask_port))
                self.mask_client.setblocking(False)

                logger.info(f"Successfully reconnected to server at port {self.mask_port}")
                break
            except ConnectionRefusedError:
                # Handle connection refused, possibly by waiting before retrying
                time.sleep(0.5)  # Wait a bit before retrying to avoid spamming connection attempts

            if safety_break > 10000:
                logger.error(f"Unable to reconnect to the server at port {self.mask_port}")
                break

    def connect_to_server(self):
        if self.mask_client is not None:
            self.connect_client()
            return

        self.update_args()
        pre_cmd = self.gizmo.knob('pre_cmd_knob').value() or None
        post_cmd = self.gizmo.knob('post_cmd_knob').value() or None

        thread = ExecuteThread(self.args, None, pre_cmd, post_cmd)
        thread.start()
        self.thread_list.append(thread)
        # print(self.pointer._instances)
        self.counter = 0 if self.counter == 1 else 1

        self.set_status(True, f"Running to server port {self.mask_port}")

        # self.mask_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_client()

        self.set_status(True, f"Connected to server port {self.mask_port}")
        return True

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
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("connect_to_server")')

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

        if knob.name() == 'track_01':
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
        xy_points = int(xy_points[0]), 852 - int(xy_points[1])

        data_to_send = {
            'use_xy': True,
            'frame': nuke.frame(),
            'prompts': {
                'point_coords': [xy_points],
                'point_labels': [1]
            }
        }
        logger.warning(f'data sent on knob change {data_to_send}')

        while True:
            try:
                self.mask_client.sendall(SocketServer.encode_data(data_to_send.copy()))
                break
            except ConnectionResetError:
                self.connect_client()

        file_paths = self.output_file_path.replace(f'.{self.frame_padding}.', f'.{nuke.frame():04d}.')
        self.check_multiple_files([self.get_node(f"Read1", 'Read')], [file_paths])
    # Handle other socket errors
    # try:
    # raw_data = self.mask_client.recv(4096)
    # logger.info(f"raw_data:{raw_data}")
    # masks = SocketServer.decode_data(raw_data)  # adjust buffer size as needed
    # logger.info(f"masks:{masks}")
    # except Exception as e:
    #     logger.error(e)

    # if masks:
    #     self.receive_mask(masks)
    # else:
    #     logger.debug(f'In the node >> {masks}')

    @property
    def output_file_path(self):
        output_dir_path = os.path.join(self.get_output_dir(), f'{datetime.now().strftime("%Y%m%d")}')
        output_dir_path = output_dir_path.replace('\\', '/')
        os.makedirs(output_dir_path, exist_ok=True)
        return self.add_padding(os.path.join(output_dir_path, 'mask.png'))

    def update_args(self):
        super().update_args()
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

    def receive_mask(self, masks):
        output_dir_path = os.path.join(self.get_output_dir(), f'{datetime.now().strftime("%Y%m%d")}')
        os.makedirs(output_dir_path, exist_ok=True)
        current_frame = nuke.frame()

        output_batch_list = []
        for i, mask in enumerate(reversed(masks)):
            filename = f'mask_{i}.{current_frame:04d}.png'
            filePath = os.path.join(output_dir_path, filename).replace('\\', '/')
            color = image_utils.roto_colors[i % len(image_utils.roto_colors)]
            image = image_utils.create_image(mask, color)
            image.save(filePath)
            output_batch_list.append(f'"{filePath}"')

        output_nodes = self.update_output_callback(output_batch_list)
        self.disconnect_inputs(self.mask_combine)

        skip_mask = 0
        for i, fn in enumerate(output_nodes):
            if i == 2:
                skip_mask = 1
            self.mask_combine.setInput(i + skip_mask, fn)
            knobName = 'mask_{:02d}_knob'.format(i + 1)
            mask_knob = self.gizmo.knob(knobName)
            if not mask_knob:
                continue

            mask_knob.setVisible(True)
            mask_knob.setValue(1)
            fn['disable'].setExpression(f"1 - {mask_knob.name()}")
        self.set_status(False, "Masks are created")


if __name__ == '__main__':
    # Create a NoOp node on which we'll add the knobs
    node = MagicRotoSelectorLive()
    node.showControlPanel()
