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

        self.args = args.copy()
        ports = args.pop('ports')
        if ports is None:
            logger.error('invalid ports')
            self.mask_port = SocketServer.find_available_port()
        else:
            self.mask_port = ports[0]

        self.mask_server = SocketServer(port=self.mask_port, data_handler=self.on_points_changed)
        logger.info(f"Creating server at {self.mask_port}")

        self.easyRoto = MagicRotoSelectorExecutor(args)
        self.segmenter = self.easyRoto.create_segmenter()
        self.easyRoto.load_image(args['image'])
        self.mask_server.start_accepting_clients()

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

        self.easyRoto.args_dict['prompts'] = common_utils.get_dict_type(data['prompts'])
        # self.easyRoto.load_image(r'C:/Users/mellithy/Downloads/5eaeab055785321b3858e963.jpg')
        logger.debug(f"{'*'*10}\n{data['prompts']} vs {self.easyRoto.args_dict['prompts']}")
        masks, scores, logits = self.easyRoto.predict()

        img = self.create_image_rgb(masks)
        frame = f'{data.get("frame", 1001):04d}'
        out_img = self.args['output_path'].replace('.%04d.', f'.{frame}.')
        img.save(out_img)
        logger.info(f'Images saved {out_img}')
        return masks

    @staticmethod
    def create_image_rgb(array_list):
        from PIL import Image
        import numpy as np
        if len(array_list) == 3:
            # Process each array and create an RGB image
            channels = []
            for arr in array_list:
                arr = image_utils.fill_holes_in_boolean_array(arr)
                rows, cols = arr.shape
                img = Image.new('L', (cols, rows))  # 'L' mode for grayscale
                pixels = img.load()

                for row in range(rows):
                    for col in range(cols):
                        # Set pixel intensity to 255 if True, else 0
                        pixels[col, row] = 255 if arr[row, col] else 0

                channels.append(img)

            return Image.merge("RGB", channels)

        elif len(array_list) == 1:
            # Convert the array to uint8 and create a grayscale image
            arr = image_utils.fill_holes_in_boolean_array(array_list[0])
            return Image.fromarray((255 * np.clip(arr, 0, 1)).astype('uint8'))

        else:
            raise ValueError("Array list must contain 1 or 3 arrays.")


if __name__ == '__main__':
    args = MagicRotoSelectorLive.setup_parser()
    args_dict = vars(args)

    try:
        lvl = int(args_dict.get('logger_level'))
    except TypeError:
        lvl = 20

    args_dict['image'] = r'C:/Users/mellithy/Downloads/5eaeab055785321b3858e963.jpg'
    logger.setLevel(lvl)

    logger.info(f"Pars args {args_dict}")

    mrSelectorLive = MagicRotoSelectorLive(args=args_dict)
    # Main loop to keep the script running
    try:
        while True:
            # Sleep to prevent the loop from consuming CPU resources
            logger.debug(f'mrselectorlive still running {mrSelectorLive.mask_server.conn}')
            time.sleep(10)
    except KeyboardInterrupt:
        # Handle graceful shutdown on interrupt (Ctrl+C)
        logger.info("Shutting down server...")
        # Implement any necessary cleanup or shutdown procedures here