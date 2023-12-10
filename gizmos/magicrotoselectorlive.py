import glob
import os
import threading
import time
from datetime import datetime

import nuke
from PIL import Image

from magicroto.config.config_utils import mg_selector_live_path
from magicroto.gizmos.core.gizmobase import GizmoBase
from magicroto.utils import common_utils
from magicroto.utils.icons import Icons
from magicroto.utils.execute_thread import ExecuteThread

from magicroto.utils.logger import logger, logger_level
from magicroto.utils.soketserver import SocketServer


class MagicRotoSelectorLive(GizmoBase):

    def __init__(self, node=None, name=None):
        super().__init__(gizmo=node, name=name)

        self.knob_change_timer = None
        self.last_knob_value = None

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
        nuke.addOnDestroy(self.on_destroy)

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

    def create_generate_knobs(self):

        self.create_generate_tab()
        super().create_generate_knobs()

        # region frame rang
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

        # if not self.gizmo.knob('use_frame_range_knobs'):
        #     use_frame_range_knobs = nuke.Boolean_Knob('use_frame_range_knobs',
        #                                               f'{Icons.video_symbol} Use frame range')
        #     use_frame_range_knobs.clearFlag(nuke.STARTLINE)
        #     self.gizmo.addKnob(use_frame_range_knobs)
        # endregion
        self.add_divider('Trackers')

        if not self.gizmo.knob('track_01'):
            track_01_xy = nuke.XY_Knob('track_01', f'Track 01')
            track_01_xy.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(track_01_xy)

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

        channels = ['red', 'green', 'blue']
        for i in range(1, 4):
            knobLabel = '{:02d}'.format(i)
            knobLabel = 'RGB'[i-1]
            knobName = f'mask_{knobLabel}_knob'
            msk_knob = self.gizmo.knob(knobName)
            if not msk_knob:
                msk_knob = nuke.Boolean_Knob(knobName, knobLabel)
                self.gizmo.addKnob(msk_knob)
                if i == 1 or i % 10 == 1:
                    msk_knob.setFlag(nuke.STARTLINE)
                msk_knob.setValue(1)

            grade_node['multiply'].setExpression(f"{knobName}", i-1)

        self.reload_button()

        status_bar = self.status_bar
        self.set_status(running=False)

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
                break

        y_shift = Image.open(existing_file).height

        xy_points = int(xy_points[0]), y_shift - int(xy_points[1])

        data_to_send = {
            'use_xy': True,
            'output_path': self.output_file_path,
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
                logger.debug(f'data_to_send: {data_to_send}')
                break

            except ConnectionResetError:
                logger.debug(f'ConnectionResetError trying ensure_server_connection at port {self.main_port}')
                self.ensure_server_connection()

            except AttributeError:
                logger.debug(f'AttributeError trying ensure_server_connection at port {self.main_port}')
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
