import inspect
import os
import time
import sys

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


from magicroto.executors.magicrotoselectorexecutor import MagicRotoSelectorExecutor
from magicroto.utils.soketserver import SocketServer
from magicroto.utils import common_utils, image_utils
import argparse

import json

from magicroto.utils.logger import logger


class MagicRotoSelectorLive:

    def __init__(self, args):

        self.current_frame = None
        self.args = args.copy()
        ports = args.pop('ports')
        if ports is None:
            logger.error('invalid ports')
            self.mask_port = SocketServer.find_available_port()
        else:
            self.mask_port = ports[0]

        logger.info(f"Creating server at {self.mask_port}")
        self.mask_server = SocketServer(port=self.mask_port, data_handler=self.on_points_changed)
        self.mask_server.start_accepting_clients()
        print(f"Creating server at {self.mask_port}")
        self.easyRoto = MagicRotoSelectorExecutor(args)
        self.segmenter = self.easyRoto.create_segmenter()
        if args['image']:
            self.easyRoto.load_image(args['image'])
        logger.info(f"Ready to use server at {self.mask_port}")

    @classmethod
    def setup_parser(cls):
        parser = argparse.ArgumentParser(description="EasyRoto tool")
        parser.add_argument("--image", type=str, required=False, help="Path to the image file.")
        parser.add_argument("--ports", type=json.loads, required=True, help="list of 2 Server ports.")
        parser.add_argument("--SAM_checkpoint", type=str, required=True, help="Path to the SAM checkpoint.")
        parser.add_argument("--cache_dir", type=str, help="Path to the models.")
        parser.add_argument("--output_path", type=str, help="Path to output image.")
        parser.add_argument("--model_type", type=str, default="vit_h", help="Type of model. Default is vit_h.")
        parser.add_argument("--device", type=str, default="cuda:0",
                            help="Device to run the model on. Default is cuda:0.")
        parser.add_argument("--mask_input", type=str, default="", help="path for a mask.")
        parser.add_argument('--logger_level', required=False, help='the level of the logger', type=int)
        parser.add_argument("--prompts", type=str, default={}, help="preset prompts.")
        args = parser.parse_args()
        return args

    def on_points_changed(self, data):
        for frame, d in data.items():
            self._on_points_changed(d, frame)

    def _on_points_changed(self, data, frame=None):
        frame = frame or f'{data.get("frame", 1001):04d}'

        self.args['output_path'] = data.get('output_path', self.args['output_path'])
        init_img_path = data.get('init_img_path', None)
        logger.debug(f'init_img_path:{init_img_path}')
        if init_img_path is not None:
            init_img_path = init_img_path.replace('.%04d.', f'.{frame:04d}.')
            logger.debug(f'renamed img << {init_img_path}')
            if init_img_path and os.path.isfile(init_img_path):
                if not self.current_frame or self.current_frame != frame:
                    self.easyRoto.load_image(init_img_path)
                    self.current_frame = frame
                    # time.sleep(1)

        self.easyRoto.args_dict['prompts'] = common_utils.get_dict_type(data['prompts'])
        result = self.easyRoto.predict()
        if result is None:
            logger.debug(f"Predict return none, server might not be ready yet")
            return

        masks, scores, logits = result
        img = image_utils.create_image_rgb(masks)

        out_img = self.args['output_path'].replace('.%04d.', f'.{frame:04d}.')
        img.save(out_img)
        logger.info(f'Images saved {out_img}')
        return masks


if __name__ == '__main__':
    args = MagicRotoSelectorLive.setup_parser()
    args_dict = vars(args)

    try:
        lvl = int(args_dict.get('logger_level'))
    except TypeError:
        lvl = 20

    # args_dict['image'] = r'C:/Users/mellithy/Downloads/5eaeab055785321b3858e963.jpg'
    logger.setLevel(lvl)

    logger.info(f"Pars args {args_dict}")

    mrSelectorLive = MagicRotoSelectorLive(args=args_dict)
    # Main loop to keep the script running
    # try:
    #     while True:
    #         # Sleep to prevent the loop from consuming CPU resources
    #         msg = mrSelectorLive.mask_server.conn if hasattr(mrSelectorLive.mask_server, 'conn') else mrSelectorLive.mask_server
    #         logger.debug(f'mrselectorlive still running {msg}')
    #         time.sleep(10)
    # except KeyboardInterrupt:
    #     # Handle graceful shutdown on interrupt (Ctrl+C)
    #     logger.info("Shutting down server...")
    #     # Implement any necessary cleanup or shutdown procedures here