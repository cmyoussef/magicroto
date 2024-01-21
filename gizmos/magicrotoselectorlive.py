import glob
import os
import threading
import time

import nuke
from PIL import Image

from magicroto.config.config_utils import mg_selector_live_path
from magicroto.gizmos.core.gizmobase import GizmoBase
from magicroto.utils import common_utils
from magicroto.utils.icons import Icons
from magicroto.utils.logger import logger
from magicroto.utils.soketserver import SocketServer


class MagicRotoSelectorLive(GizmoBase):

    def __init__(self, node=None, name=None):
        super().__init__(gizmo=node, name=name)

        self.knob_change_timer = None
        self.last_knob_value = {}

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

        display_mask_knob = self.gizmo.knob('display_mask_knob')
        self.plus_merge_node['disable'].setExpression(f"1-{display_mask_knob.name()}")

        # Clear selection
        nuke.selectAll()
        nuke.invertSelection()
        # Creating or getting the mask_combine node

        # 1. Connect the input_node to the B input of plus_mask
        self.plus_merge_node.setInput(0, self.input_node)

        # 2. Connect mask_combine to the A input of plus_mask
        self.plus_merge_node.setInput(1, self.get_node(f"Read1", 'Read'))

        self.add_gradient_and_expression()

        # 3. Connect plus_merge_node to output_node
        # self.output_node.setInput(0, self.plus_merge_node)

        # End of gizmo modification
        self.gizmo.end()

        self.points = []
        self.counter = 0

    def add_gradient_and_expression(self):
        # Clear selection
        nuke.selectAll()
        nuke.invertSelection()

        # Assuming self.get_node("Read1", 'Read') returns the Read node
        read_node = self.get_node("Read1", 'Read')

        # nuke.createNode("Grade")
        # 1. Create a Grade Node
        grade_node = self.get_node("Channel_selector", 'Grade')
        grade_node.setInput(0, read_node)
        # grade_node['multiply'].setValue(0, 1)
        grade_node['multiply'].setSingleValue(False)

        # 2. Create an Expression Node
        expression_node = self.get_node("alpha_convert", 'Expression')
        expression_node.setInput(0, grade_node)
        expression_node['expr3'].setValue("clamp(r+g+b, 0, 1)")

        # 3. Create a Copy Node
        copy_node = self.get_node("copy_alpha", 'Copy')
        copy_node.setInput(0, self.plus_merge_node)  # Existing node connected to the output
        copy_node.setInput(1, expression_node)
        copy_node['from0'].setValue("alpha")
        copy_node['to0'].setValue("alpha")

        # Update the output node connection to use the Copy node
        self.output_node.setInput(0, copy_node)
        self.plus_merge_node.setInput(1, grade_node)

        return grade_node

    def write_input(self):
        if not self.is_connected(self.input_node):
            return

        init_img_path = self.get_init_img_path()
        init_img_path_padding = self.add_padding(init_img_path)
        existing_file = init_img_path_padding.replace(f'.{self.frame_padding}.', f".{nuke.frame():04d}.")

        self.writeInput(init_img_path, self.input_node)
        return existing_file

    @property
    def input_node(self):
        return self.get_node("image", 'Input')

    @property
    def mask_input_node(self):
        return self.get_node("mask", 'Input')

    def create_tracking_knob(self, n, neg=False):
        tracking_label = f'track_{n:02d}'
        if neg:
            tracking_label += '_neg'

        knobName = tracking_label
        if not self.gizmo.knob(knobName):
            track_xy = nuke.XY_Knob(knobName, tracking_label)
            track_xy.setFlag(nuke.STARTLINE)
            if n != 1 or neg:
                track_xy.setEnabled(False)
            self.gizmo.addKnob(track_xy)

        knobName = tracking_label + '_enable'

        if not self.gizmo.knob(knobName):
            track_check = nuke.Boolean_Knob(knobName, '')
            # track_xy.setFlag(nuke.STARTLINE)
            if n == 1 and not neg:
                track_check.setValue(True)
            track_check.clearFlag(nuke.STARTLINE)
            self.gizmo.addKnob(track_check)

    def create_generate_knobs(self):

        self.create_generate_tab()
        super().create_generate_knobs()

        self.add_divider('Trackers')

        for i in range(1, 4):
            self.create_tracking_knob(i)
        self.add_divider()
        for i in range(1, 4):
            self.create_tracking_knob(i, True)

        self.add_divider('masks')

        display_mask_knob = self.gizmo.knob('display_mask_knob')
        if not display_mask_knob:
            display_mask_knob = nuke.Boolean_Knob('display_mask_knob', 'display mask')
            display_mask_knob.setFlag(nuke.STARTLINE)
            display_mask_knob.setValue(True)
            self.gizmo.addKnob(display_mask_knob)

        grade_node = self.get_node("Channel_selector", 'Grade')
        grade_node['multiply'].setSingleValue(False)
        self._force_evaluate_node(grade_node)

        for i in range(1, 4):
            knobLabel = '{:02d}'.format(i)
            knobLabel = 'RGB'[i - 1]
            knobName = f'mask_{knobLabel}_knob'
            msk_knob = self.gizmo.knob(knobName)
            if not msk_knob:
                msk_knob = nuke.Boolean_Knob(knobName, knobLabel)
                self.gizmo.addKnob(msk_knob)
                if i == 1 or i % 10 == 1:
                    msk_knob.setFlag(nuke.STARTLINE)
                msk_knob.setValue(1)

            grade_node['multiply'].setExpression(f"{knobName}", i - 1)

        # region frame range
        self.add_divider('Frame Range')

        self.create_frame_range_knobs()

        if not self.gizmo.knob('keys_knob'):
            keys_knob = nuke.Int_Knob('keys_knob', f'Keys {Icons.key_symbol}')
            keys_knob.setFlag(nuke.STARTLINE)
            # use_external_execute.setFlag(nuke.DISABLED)
            self.gizmo.addKnob(keys_knob)

        if not self.gizmo.knob('cache_frame_btn_knob'):
            cn_button = nuke.PyScript_Knob('cache_frame_btn_knob', f'Cache input {Icons.download_symbol}')
            cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('cache_frame_btn_knob').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("cache_input")')

        if not self.gizmo.knob('segment_sequence_btn_knob'):
            cn_button = nuke.PyScript_Knob('segment_sequence_btn_knob', f'Segment Sequence {Icons.execute_symbol}')
            cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('segment_sequence_btn_knob').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("segment_sequence")')
        # endregion

        # if not self.gizmo.knob('use_frame_range_knobs'):
        #     use_frame_range_knobs = nuke.Boolean_Knob('use_frame_range_knobs',
        #                                               f'{Icons.video_symbol} Use frame range')
        #     use_frame_range_knobs.clearFlag(nuke.STARTLINE)
        #     self.gizmo.addKnob(use_frame_range_knobs)
        # endregion

        self.reload_button()

        status_bar = self.status_bar
        self.set_status(running=False)


    def get_prompt_data(self):

        prompt_data = {}
        for knob_name, knob in self.gizmo.knobs().items():
            # Check if 'track_' is in the knob name
            if 'track_' in knob_name and not knob_name.endswith('_enable'):
                if not knob.enabled():
                    continue
                xy = int(knob.value()[0]), int(knob.value()[1])
                prompt_data[knob_name] = (xy, not knob_name.endswith('_neg'))
        return prompt_data

    def segment_sequence(self):
        init_img_path = self.get_init_img_path()
        init_img_path_padding = self.add_padding(init_img_path)
        multi_frame_data = {}
        start_frame, end_frame = self.frame_range
        logger.debug(f'self.frame_range {self.frame_range}')

        # set keyFrames
        keys = self.gizmo.knob('keys_knob')
        # Make sure the knob is set to be animated
        keys.setAnimated()

        for i in range(start_frame, end_frame + 1):
            # nuke.frame(i)
            # nuke.execute(self.gizmo, i, i)
            # nuke.executeInMainThreadWithResult(nuke.frame, args=(i, ))
            existing_file = init_img_path_padding.replace(f'.{self.frame_padding}.', f".{i:04d}.")
            # nuke.executeInMainThread(nuke.frame, args=(i, ))
            self.force_evaluate_nodes()
            logger.debug(f"Move to frame {i}")
            if not os.path.exists(existing_file):
                self.writeInput(init_img_path, self.input_node, frame_range=(i, i))
            else:
                self.writeInput(init_img_path, self.input_node, frame_range=(i, i), temp=True)

            y_shift = Image.open(existing_file).height
            prompt_data = self.get_prompt_data()
            logger.debug(f'prompt_data:{prompt_data}')
            prompts = {}
            for k, (point_coord, point_label) in prompt_data.items():
                point_coord = int(point_coord[0]), y_shift - int(point_coord[1])
                prompts.setdefault('point_coords', []).append(point_coord)
                prompts.setdefault('point_labels', []).append(1 if point_label else 0)
            frame_data = {
                'use_xy': True,
                'output_path': self.output_file_path,
                'frame': i,
                'init_img_path': init_img_path_padding,
                'prompts': prompts
            }

            multi_frame_data[i] = frame_data
            keys.setValueAt(i, i)

        self.attempt_to_send(multi_frame_data)

        file_paths = self.output_file_path.replace(f'.{self.frame_padding}.', f'.{nuke.frame():04d}.')
        self.check_multiple_files([self.get_node(f"Read1", 'Read')], [file_paths])

    def cache_input(self):
        init_img_path = self.get_init_img_path()
        self.writeInput(init_img_path, self.input_node, frame_range=self.frame_range)
        logger.info(f"cache images updates, {init_img_path}")

    def check_last_knob_value(self, xy, knob_name):
        if knob_name not in self.last_knob_value:
            return True
        else:
            return xy != self.last_knob_value[knob_name][0]

    def knobChanged(self, knob=None):

        knob = knob or nuke.thisKnob()
        send_call = [False]
        name = knob.name()
        if 'track_' in name and name.endswith('_enable'):
            track_knob = self.gizmo.knob(name[:-len('_enable')])
            track_knob.setEnabled(knob.value())

        if 'track_' in name and self.mask_client and not name.endswith('_enable'):

            if knob.enabled():
                xy = int(knob.value()[0]), int(knob.value()[1])
                # Check if the value has changed
                if self.check_last_knob_value(xy, name):
                    # Update the last known value
                    self.last_knob_value[name] = (xy, not name.endswith('_neg'))
                    send_call.append(True)

        if any(send_call):
            # Cancel the existing timer if it's running
            if self.knob_change_timer is not None:
                self.knob_change_timer.cancel()

            # Create a new timer
            self.knob_change_timer = threading.Timer(0.5, self.on_points_changed)
            self.knob_change_timer.start()

        super().knobChanged(knob)

    def on_points_changed(self):

        init_img_path = self.get_init_img_path()
        init_img_path_padding = self.add_padding(init_img_path)
        existing_file = init_img_path_padding.replace(f'.{self.frame_padding}.', f".{nuke.frame():04d}.")
        logger.debug(f'{existing_file}')
        if not os.path.exists(existing_file):
            # self.writeInput(init_img_path, self.input_node)
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
                break

        y_shift = Image.open(existing_file).height
        prompts = {}
        for k, (point_coord, point_label) in self.get_prompt_data().items():
            point_coord = int(point_coord[0]), y_shift - int(point_coord[1])
            prompts.setdefault('point_coords', []).append(point_coord)
            prompts.setdefault('point_labels', []).append(1 if point_label else 0)

        data_to_send = {
            'use_xy': True,
            'output_path': self.output_file_path,
            'frame': nuke.frame(),
            'init_img_path': init_img_path_padding,
            'prompts': prompts
        }
        multiData = {nuke.frame(): data_to_send}
        self.attempt_to_send(multiData)

        file_paths = self.output_file_path.replace(f'.{self.frame_padding}.', f'.{nuke.frame():04d}.')
        self.check_multiple_files([self.get_node(f"Read1", 'Read')], [file_paths])

    def _attempt_to_send(self, data_to_send):
        if self.knob_change_timer is not None:
            self.knob_change_timer.cancel()

        # Create a new timer
        self.knob_change_timer = threading.Timer(0.5, self._attempt_to_send, args=(data_to_send,))
        self.knob_change_timer.start()

    def attempt_to_send(self, data_to_send):
        try_count = 0
        while True and try_count < 11:
            try_count += 1

            if try_count == 10:
                logger.warning(f"Failed to send to the server at {self.main_port}")

            try:
                self.mask_client.sendall(SocketServer.encode_data(data_to_send.copy()))
                logger.debug(f'data_to_send: {data_to_send}')
                break

            except ConnectionResetError:
                logger.debug(f'ConnectionResetError trying ensure_server_connection at port {self.main_port}')
                self.ensure_server_connection()

            except AttributeError:
                logger.debug(f'AttributeError trying ensure_server_connection at port {self.main_port}')
                self.ensure_server_connection()

    def update_args(self):
        super().update_args()
        img_path = self.write_input()
        if img_path:
            self.args['image'] = img_path
        self.args['output_path'] = self.output_file_path
        self.args['script_path'] = mg_selector_live_path
        self.args['SAM_checkpoint'] = os.path.join(self.args.get('cache_dir'), 'sam_vit_h_4b8939.pth')

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


if __name__ == '__main__':
    # Create a NoOp node on which we'll add the knobs
    node = MagicRotoSelectorLive()
    node.showControlPanel()
