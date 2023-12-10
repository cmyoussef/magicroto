import argparse
import json
import os.path
from typing import Optional

import numpy as np
from PIL import Image

from magicroto.core.base_segmenter import BaseSegmenter
from magicroto.utils import common_utils
from magicroto.utils.logger import logger


class MagicRotoSelectorExecutor:
    def __init__(self, args=None):
        self.parser: Optional[argparse.ArgumentParser] = None
        # self.setup_parser()
        logger.setLevel(int(args.pop('logger_level')) or 20)
        self.args_dict: Optional[dict] = args or {}
        self.segmenter: Optional[BaseSegmenter] = None
        self.image: Optional[Image] = None
        self.masks: Optional[list] = None
        self.mask: Optional[Image] = None
        self.refine_args()

        # self.load_mask()
        if 'mask_input' in args:
            self.load_mask(args.get('mask_input'))

        mode = 'point'
        if self.mask is not None and self.is_points():
            mode = 'both'
        self.args_dict['mode'] = mode

        logger.debug(f" self.args_dict, {self.args_dict}")

    def setup_parser(self):
        self.parser = argparse.ArgumentParser(description="Magic Roto Selector")
        self.parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
        self.parser.add_argument("--SAM_checkpoint", type=str, required=True, help="Path to the SAM checkpoint.")
        self.parser.add_argument("--prompts", type=json.loads, required=True,
                                 help="JSON string representing the prompts dictionary.")
        self.parser.add_argument("--model_type", type=str, default="vit_h", help="Type of model. Default is vit_h.")
        self.parser.add_argument("--device", type=str, default="cuda:0",
                                 help="Device to run the model on. Default is cuda:0.")
        self.parser.add_argument("--mask_input", type=str, default="", help="path for a mask.")
        self.parser.add_argument("--output", type=str, required=True, help="output directory.")
        self.args = self.parser.parse_args()
        self.args_dict = vars(self.args)

    def create_segmenter(self):
        self.segmenter = BaseSegmenter(SAM_checkpoint=self.args_dict['SAM_checkpoint'],
                                       model_type=self.args_dict['model_type'],
                                       device=self.args_dict['device'])
        return self.segmenter

    def load_image(self, image_path=None):
        image_path = image_path or self.args_dict['image']
        self.args_dict['image'] = image_path
        img = Image.open(image_path).convert("RGB")
        # img = self.ensure_BCHW(img)
        self.image = img
        if self.segmenter is None:
            logger.warning(
                f"Segmenter is not initialized yet, you might need to wait a few moments, or restart the server")
            return
        self.segmenter.reset_image()
        self.segmenter.set_image(np.array(self.image))
        logger.info(f'Setting Image << {image_path}, {self.image}')
        return self.image

    def load_mask(self, mask_path=None):

        if mask_path is None and isinstance(self.args_dict['mask_input'], str):
            return

        mask_path = mask_path
        if os.path.exists(mask_path):
            image = Image.open(mask_path)

            # Check if the image has an alpha channel
            if image.mode == 'RGBA':
                # Extract the alpha channel
                alpha = image.split()[-1]
            else:
                # Extract the red channel
                alpha = image.getchannel('R')

            # Convert the channel to a numpy array
            self.mask = np.array(alpha)
            self.args_dict['prompts']['mask_input'] = self.mask[None, :, :]
        # else:
        #     logger.error(f'Mask Path dose not exists \n{mask_path}')

    def predict(self):
        if self.segmenter is None:
            logger.warning(
                f"Segmenter is not initialized yet, you might need to wait a few moments, or restart the server")
            return

        if not self.is_points:
            return

        self.refine_args()

        if self.mask is not None or self.is_points():
            self.masks, scores, logits = self.segmenter.predict(self.args_dict['prompts'],
                                                                self.args_dict['mode'])
            return self.masks, scores, logits

    def is_points(self):
        if 'prompts' in self.args_dict:
            if 'point_coords' in self.args_dict['prompts']:
                return True
        return False

    def refine_args(self):
        if self.is_points():
            self.args_dict['prompts'] = common_utils.get_dict_type(self.args_dict['prompts'])
            self.args_dict['prompts']['point_coords'] = np.array(self.args_dict['prompts']['point_coords'])


if __name__ == "__main__":
    easyRoto = MagicRotoSelectorExecutor()
    logger.info(f'Initialize the EasyRoto')

    easyRoto.setup_parser()
    logger.info(f'parsing args')

    easyRoto.create_segmenter()
    logger.info(f'Create segmenter: {easyRoto.segmenter.__class__.__name__}')

    easyRoto.predict()
    logger.info(f'Output is generated.')
