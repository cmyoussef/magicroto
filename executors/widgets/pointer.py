import argparse
import os
import inspect
import json
import socket
import sys

import numpy as np
from PySide2 import QtWidgets, QtCore

# TODO: if it's installed with envVar you do not need that
# <editor-fold desc="append module">
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
base_name = os.path.basename(current_dir)
module_dir = os.path.dirname(current_dir)
safe_brake = 0
while safe_brake < 100:
    base_name = os.path.basename(module_dir)
    module_dir = os.path.dirname(module_dir)
    if base_name == 'magicroto':
        break
    safe_brake += 1

paths = [module_dir]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)
# </editor-fold>

from magicroto.config.config_utils import config_dict
from magicroto.config.config_utils import easy_roto_path
from magicroto.executors.magicrotoselectorexecutor import MagicRotoSelectorExecutor
from magicroto.gizmos.widgets.pointer import ZoomablePointer
from magicroto.utils.soketserver import SocketServer
from magicroto.utils.logger import logger
from magicroto.utils import common_utils


class PointerWindow(QtWidgets.QMainWindow):
    def __init__(self, node=None, parent=None, args=None, ports=None):
        super(PointerWindow, self).__init__(parent)

        if ports is None:
            logger.error('invalid ports')
            self.mask_port = SocketServer.find_available_port()
            self.pointer_data_port = SocketServer.find_available_port()
        else:
            self.mask_port = ports[0]
            self.pointer_data_port = ports[1]

        self.mask_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mask_client.connect(("localhost", self.mask_port))

        self.pointer_data_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.pointer_data_client.connect(("localhost", self.pointer_data_port))

        self.setWindowTitle("Magic Roto Selector")
        self.setGeometry(100, 100, 800, 600)

        self.easyRoto = MagicRotoSelectorExecutor(args)
        self.segmenter = self.easyRoto.create_segmenter()

        # Initialize your Pointer class and set it as the central widget
        _prompts = args_dict.get('prompts', {})
        _prompts = common_utils.get_dict_type(_prompts)
        initial_points = _prompts.get('point_coords', None)
        initial_labels = _prompts.get('point_labels', None)
        self.pointer = ZoomablePointer(node, initial_points, initial_labels, easyRoto=self.easyRoto)
        self.send_mask_btn = QtWidgets.QPushButton('Send to Nuke')

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.pointer)

        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.send_mask_btn)

        self.layout.addLayout(self.hbox)

        self.central_widget = QtWidgets.QWidget()
        self.central_widget.setLayout(self.layout)

        self.setCentralWidget(self.central_widget)

        # Connect signals
        self.send_mask_btn.clicked.connect(self.send_mask)
        # self.send_mask_btn.clicked.connect(self.close)

        # Connect signal
        self.pointer.connect_signal(self.on_points_changed)
        self.mask = []

        if not args:
            return

        if 'image' in args:
            self.pointer.load_image(args.get('image'))

        # if 'prompts' in args or self.pointer.mask:
        #     prompts = args.get('prompts', {})
        #     prompts = common_utils.get_dict_type(prompts)
        #     logger.warning(f'{type(prompts)}, {prompts}')
        #     self.on_points_changed(prompts)

    @classmethod
    def setup_parser(cls):
        parser = argparse.ArgumentParser(description="EasyRoto tool")
        parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
        parser.add_argument("--ports", type=json.loads, required=True, help="list of 2 Server ports.")
        parser.add_argument("--SAM_checkpoint", type=str, required=True, help="Path to the SAM checkpoint.")
        parser.add_argument("--cache_dir", type=str, help="Path to the models.")
        parser.add_argument("--model_type", type=str, default="vit_h", help="Type of model. Default is vit_h.")
        parser.add_argument("--device", type=str, default="cuda:0",
                            help="Device to run the model on. Default is cuda:0.")
        parser.add_argument("--mask_input", type=str, default="", help="path for a mask.")
        parser.add_argument('--logger_level', required=False, help='the level of the logger', type=int)
        parser.add_argument("--prompts", type=str, default={}, help="preset prompts.")
        args = parser.parse_args()
        return args

    @QtCore.Slot()
    def on_points_changed(self, data):
        points = data.get('point_coords', [])
        point_labels = data.get('point_labels', [])

        prompts = {
            'point_coords': points,
            'point_labels': point_labels
        }
        if not points:
            self.pointer.clear()
            return

        self.easyRoto.args_dict['prompts'] = common_utils.get_dict_type(self.easyRoto.args_dict['prompts'])
        self.pointer_data_client.sendall(SocketServer.encode_data(prompts))
        self.easyRoto.args_dict['prompts'].update(prompts)

        masks, scores, logits = self.easyRoto.predict()
        self.overlay_masks(masks)
        ack_data = self.pointer_data_client.recv(1024)  # adjust buffer size as needed

    def overlay_masks(self, masks):
        self.mask = [np.array(m, dtype=np.bool_) for m in masks]
        self.pointer.overlay_masks(masks)

    @QtCore.Slot()
    def send_mask(self):
        # Send the serialized numpy array
        self.mask_client.sendall(SocketServer.encode_data(self.mask))
        ack_data = self.mask_client.recv(1024)  # adjust buffer size as needed
        self.close()
        # time.sleep(1)

    def closeEvent(self, event) -> None:
        header = b'command::'
        self.mask_client.sendall(header + b'quit')
        ack_data = self.mask_client.recv(1024)  # adjust buffer size as needed
        self.pointer_data_client.sendall(header + b'quit')
        ack_data = self.pointer_data_client.recv(1024)  # adjust buffer size as needed
        # if ack_data:
        super().closeEvent(event)

    def show(self):
        super(PointerWindow, self).show()

    @classmethod
    def refine_args(cls, inDict):
        if 'prompts' in inDict:
            inDict = common_utils.get_dict_type(args['prompts'].replace("'", '"'))

        return inDict


def test_run():
    app = QtWidgets.QApplication([])

    in_args = {'python_exe': config_dict.get('python_exe'),
               'cache_dir': config_dict.get('cache_dir'),
               'SAM_checkpoint': r'D:/track_anything_project/sam_vit_h_4b8939.pth',
               'model_type': 'vit_h',
               'device': "cuda:0",
               'mode': 'point',
               'script_path': easy_roto_path}
    node = None  # Set to your actual node
    mainWin = PointerWindow(node, args=in_args)
    img = r'C:\Users\mellithy\easyTrack\ET_EasyRoto\source_EasyRoto\init_img.png'
    img = r'C:/Users/mellithy/easyTrack/SD_Txt2Img/models--kandinsky-community--kandinsky-2-1/20231021_121816/20231021_121816_1_1.png'
    mainWin.pointer.loadImage(img)
    mainWin.show()
    app.exec_()


# How to show the window
if __name__ == '__main__':
    # test_run()
    app = QtWidgets.QApplication([])

    args = PointerWindow.setup_parser()
    args_dict = vars(args)
    ports = args_dict.pop('ports')

    try:
        lvl = int(args_dict.get('logger_level'))
    except TypeError:
        lvl = 20

    logger.setLevel(lvl)
    # prompts = args_dict.get('prompts', {})
    # prompts = common_utils.get_dict_type(prompts)
    # args_dict['args'] = prompts
    # logger.info(f"Pars args {type(prompts)}")
    # logger.info(f"Pars args {args_dict}")

    easy_roto_window = PointerWindow(args=args_dict, ports=ports)

    easy_roto_window.show()
    app.exec_()
