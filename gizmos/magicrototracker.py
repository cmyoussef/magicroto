import os
from datetime import datetime

import nuke

from magicroto.config.config_utils import mg_tracking_path
from magicroto.gizmos.core.gizmobase import GizmoBase
from magicroto.utils.icons import Icons
from magicroto.utils.logger import logger_level, logger


class MagicRotoTracker(GizmoBase):
    @property
    def input_node(self):
        return self.get_node("image", 'Input')

    @property
    def mask_node(self):
        return self.get_node("mask", 'Input')

    def __init__(self, node=None, name=None):
        super().__init__(gizmo=node, name=name)

        self.knob_change_timer = None
        self.last_knob_value = {}

        # Variable to store the image
        if not self._initialized:
            self.is_client_connected = False  # Flag to track connection status

            self.args['script_path'] = mg_tracking_path
            self.populate_ui()

        self.args = {'python_exe': self.args.get('python_exe'),
                     'cache_dir': self.args.get('cache_dir'),
                     'XMEM_checkpoint': os.path.join(self.args.get('cache_dir'), 'XMem-s012.pth'),
                     'SAM_checkpoint': os.path.join(self.args.get('cache_dir'), 'sam_vit_h_4b8939.pth'),
                     'device': "cuda:0",
                     'script_path': mg_tracking_path}

        self.gizmo.begin()
        _ = self.input_node
        _ = self.mask_node
        read_node = self.get_node(f"Read{1}", 'Read')
        self.output_node.setInput(0, read_node)

        # End of gizmo modification
        self.gizmo.end()

    def update_args(self):
        self.args['logger_level'] = logger_level.get(self.gizmo.knob('logger_level_menu').value(), 20)
        self.args['python_exe'] = self.python_path
        self.args['cache_dir'] = self.cache_dir

        start_frame, end_frame = self.frame_range
        self.args['start_frame'] = start_frame
        self.args['end_frame'] = end_frame

        image_path = self.add_padding(self.get_init_img_path())
        self.writeInput(image_path, self.input_node, frame_range=(start_frame, end_frame))
        self.args['image_path'] = image_path

        self.args['ref_frames'] = self.get_key_frames()

        mask_path = self.add_padding(self.get_init_img_path('mask'))
        self.writeInput(mask_path, self.mask_node, frame_range=None)
        self.args['mask_path'] = mask_path
        self.args['mask_frames'] = [self.get_key_frames()]
        self.args['output'] = self.output_file_path

    def on_execute(self):
        read_node = self.get_node(f"Read{1}", 'Read')
        super().on_execute()
        logger.debug(f"{read_node}, {self.args['output']}")
        self.check_multiple_files(read_node, self.args['output'])

    def get_key_frames(self):
        node = self.gizmo.input(1)
        return super().get_keyFrames(node)

    @property
    def output_file_path(self):
        file_name = f'track.png'
        output_dir_path = os.path.join(self.get_output_dir(), f'{datetime.now().strftime("%Y%m%d")}')
        output_dir_path = output_dir_path.replace('\\', '/')
        os.makedirs(output_dir_path, exist_ok=True)
        return self.add_padding(os.path.join(output_dir_path, file_name))

    def create_generate_knobs(self):
        self.create_generate_tab()

        self.add_divider('frame_range')

        self.create_frame_range_knobs()

        self.add_divider()

        self.create_execute_buttons()

        self.reload_button()

        if not self.gizmo.knob('interrupt_btn'):
            interrupt_btn = nuke.PyScript_Knob('interrupt_btn', f'Force terminate All{Icons.explosion_symbol}')
            interrupt_btn.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(interrupt_btn)
        self.gizmo.knob('interrupt_btn').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("on_interrupt")')

        status_bar = self.status_bar
        self.set_status(running=False)
        # super().create_generate_knobs()


if __name__ == '__main__':
    # Create a NoOp node on which we'll add the knobs
    node = MagicRotoTracker()
    node.showControlPanel()
