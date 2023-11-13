import os
from datetime import datetime
import glob


import nuke

from magicroto.config.config_utils import easy_roto_path, easy_roto_gui_path
from magicroto.gizmos.core.gizmobase import GizmoBase
from magicroto.utils import image_utils
from magicroto.utils import common_utils
from magicroto.utils.logger import logger
from magicroto.utils.execute_thread import ExecuteThread
# from easytrack.gizmos.widgets.pointer import Pointer
from magicroto.utils.icons import Icons
from magicroto.utils.soketserver import SocketServer


def dummy_function(x, y):
    print(f"Dummy function called. Coordinates: x={x}, y={y}")


class MagicRotoSelector(GizmoBase):

    def __init__(self, node=None, name=None):
        super().__init__(gizmo=node, name=name)
        self.mask_port = SocketServer.find_available_port()
        self.pointer_data_port = SocketServer.find_available_port()
        # Variable to store the image
        if not self._initialized:
            self.mask_server = SocketServer(port=self.mask_port, data_handler=self.receive_mask)
            self.pointer_data_server = SocketServer(port=self.pointer_data_port, data_handler=self.receive_points_data)

            self.args['script_path'] = easy_roto_path
            self.populate_ui()

        self.pointer_gui_args = {'python_exe': self.args.get('python_path'),
                                 'cache_dir': self.args.get('cache_dir'),
                                 'SAM_checkpoint': r'/jobs/ADGRE/ldev_pipe/nuke/Track-Anything/trackAnyThing/Track-Anything/checkpoints/sam_vit_h_4b8939.pth',
                                 'model_type': 'vit_h',
                                 'device': "cuda:0",
                                 'script_path': easy_roto_gui_path}
        self.gizmo.begin()
        # Creating or getting the plus_mask node and setting its operation to "plus"
        self.plus_merge_node = self.get_node("plus_mask", 'Merge2')
        self.plus_merge_node['operation'].setValue("over")
        self.plus_merge_node['mix'].setValue(.5)
        # Clear selection
        nuke.selectAll()
        nuke.invertSelection()
        # Creating or getting the mask_combine node
        self.mask_combine = self.get_node("mask_combine", 'Merge2')

        # Explicitly setting connections to make sure they are as you expect

        # 1. Connect the input_node to the B input of plus_mask
        self.plus_merge_node.setInput(0, self.input_node)

        # 2. Connect mask_combine to the A input of plus_mask
        self.plus_merge_node.setInput(1, self.mask_combine)

        # 3. Connect plus_merge_node to output_node
        self.output_node.setInput(0, self.plus_merge_node)

        # self.mask_input_node

        # End of gizmo modification
        self.gizmo.end()

        self.points = []
        self.counter = 0

    @property
    def input_node(self):
        return self.get_node("image", 'Input')

    @property
    def mask_input_node(self):
        return self.get_node("mask", 'Input')

    def create_generate_knobs(self):
        self.create_generate_tab()
        if not self.gizmo.knob('load_img_btn'):
            cn_button = nuke.PyScript_Knob('load_img_btn', f'Launch UI {Icons.launch_gui_symbol}')
            cn_button.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(cn_button)
        self.gizmo.knob('load_img_btn').setCommand(
            f'import {self.base_class};{self.__class__.__module__}.{self.__class__.__name__}.run_instance_method("load_img")')

        self.add_divider('masks')

        for i in range(1, 51):
            knobLabel = '{:02d}'.format(i)
            knobName = f'mask_{knobLabel}_knob'
            if not self.gizmo.knob(knobName):
                msk_knob = nuke.Boolean_Knob(knobName, knobLabel)
                self.gizmo.addKnob(msk_knob)
                if i == 1 or i % 10 == 1:
                    msk_knob.setFlag(nuke.STARTLINE)
                msk_knob.setVisible(False)
                msk_knob.setValue(1)

        if not self.gizmo.knob('keys_knob'):
            keys_knob = nuke.Int_Knob('keys_knob', f'Keys {Icons.key_symbol}')
            keys_knob.setFlag(nuke.STARTLINE)
            # use_external_execute.setFlag(nuke.DISABLED)
            self.gizmo.addKnob(keys_knob)

        status_bar = self.status_bar
        self.set_status(running=False)
        # super().create_generate_knobs()

    def update_args(self):
        super().update_args()
        self.args['python_path'] = self.python_path
        self.args['cache_dir'] = self.cache_dir

    def load_img(self):

        if not self.gizmo.input(0):
            nuke.messege("you have to connect an image")

        init_img_path = self.get_init_img_path()
        self.writeInput(init_img_path, self.input_node)

        mask_input = None
        if self.gizmo.input(1):
            mask_input = self.get_init_img_path(img_name='mask_input')
            self.writeInput(mask_input, self.mask_input_node)


        for k, v in self.args.items():
            if k in self.pointer_gui_args:
                self.pointer_gui_args[k] = v

        pre_cmd = self.gizmo.knob('pre_cmd_knob').value() or None
        post_cmd = self.gizmo.knob('post_cmd_knob').value() or None
        self.update_args()
        
        # 'script_path': easy_roto_gui_path
        self.pointer_gui_args['image'] = init_img_path
        self.pointer_gui_args['ports'] = [self.mask_port, self.pointer_data_port]
        self.pointer_gui_args['script_path'] = easy_roto_gui_path
        # self.pointer_gui_args['python_exe'] = self.python_path
        # self.pointer_gui_args['cache_dir'] = self.cache_dir
        # if mask_input:
        # if mask_input:
        #     self.pointer_gui_args['mask_input'] = mask_input

        self.pointer_gui_args['prompts'] = common_utils.get_dict_type(self.data)

        # Accept a client connection
        self.mask_server.start_accepting_clients()
        self.pointer_data_server.start_accepting_clients()
        logger.debug(f'pre_cmd: {pre_cmd}')
        logger.debug(f'pointer_gui_args: {self.pointer_gui_args}')
        thread = ExecuteThread(self.pointer_gui_args, None, pre_cmd, post_cmd)
        thread.start()
        self.thread_list.append(thread)
        # print(self.pointer._instances)
        self.counter = 0 if self.counter == 1 else 1

        self.set_status(True, "Launching UI")

    def receive_points_data(self, data):
        nuke.executeInMainThread(lambda d=data: setattr(self, 'data', d))
        # nuke.executeInMainThread(setData, args=(data,))
        # self.data = data

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
            logger.info(f'setting {frame}')
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
    node = MagicRotoSelector()
    node.showControlPanel()
