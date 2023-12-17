import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

paths = ['D:/track_anything_project', 'D:/track_anything_project/magicroto/core']
for p in paths:
    if p not in sys.path:
        sys.path.append(p)

from magicroto.core.inpainter.base_inpainter import BaseInpainter
from magicroto.utils.logger import logger


class InPaintExecutor:
    def __init__(self, E2FGVI_checkpoint, device):
        """
        Initialize the InPaintExecutor with a BaseInpainter instance.

        @param E2FGVI_checkpoint: Path to the model checkpoint.
        @param device: Device to use for computation.
        """
        self.base_inPaint = BaseInpainter(E2FGVI_checkpoint, device)

    @staticmethod
    def setup_parser():
        """
        Set up command-line argument parsing.
        @return: ArgumentParser object.
        """
        parser = argparse.ArgumentParser(description='inPaint images using BaseInpainter.')
        parser.add_argument('--cache_dir', required=True, help='Path to the model checkpoint')
        parser.add_argument('--E2FGVI_checkpoint', required=True, help='Path to the model checkpoint')
        parser.add_argument('--device', default='cuda:0', help='Device to use for computation')
        parser.add_argument('--image_path', required=True, help='Wildcard path to the frame images')
        parser.add_argument('--mask_path', required=True, help='Wildcard path to the mask images')
        parser.add_argument('--output', required=True, help='Directory to save inpainted images')
        parser.add_argument('--ratio', type=float, default=1, help='Down-sampling ratio')
        parser.add_argument('--start_frame', type=int, default=1001, help='First frame to paint')
        parser.add_argument('--logger_level', type=int, default=1001, help='First frame to paint')
        parser.add_argument('--end_frame', type=int, default=1001, help='Last frame to paint')
        return parser

    @staticmethod
    def load_images(path, mode='RGB', frame_range=None):
        """
        Load images from a wildcard path and convert them to a numpy array.

        @param path: Wildcard path to the images.
        @param mode: Mode for converting images ('RGB' or 'P').
        @param frame_range: Frame range (1001, 1100).
        @return: Numpy array of images.
        """
        if frame_range is not None:
            images = []
            for f in range(frame_range[0], frame_range[1]+1):
                image_path = path.replace('.%04d.', f'.{f:04d}.')
                images.append(Image.open(image_path).convert(mode))

        else:
            image_paths = glob.glob(path.replace('%04d', '*'))
            image_paths.sort()
            images = [Image.open(image_path).convert(mode) for image_path in image_paths]
        logger.debug(images)
        return np.stack(images, 0)

    def execute_inpainting(self, frames, masks, output, ratio=1):
        """
        Execute inpainting on provided frames and masks.

        @param frames: Numpy array of frames.
        @param masks: Numpy array of masks.
        @param output: Directory to save inPainted images.
        @param ratio: Down-sampling ratio.
        @return: None
        """
        if len(frames) != len(masks):
            raise ValueError("The number of frames and masks must be the same.")

        inPainted_frames = self.base_inPaint.inpaint(frames, masks, ratio=ratio)

        for ti, inPainted_frame in enumerate(inPainted_frames):
            frame = Image.fromarray(inPainted_frame).convert('RGB')
            frame.save(output.replace('.%04d.', f'.{ti:04d}.'))


if __name__ == '__main__':
    args = InPaintExecutor.setup_parser().parse_args()
    executor = InPaintExecutor(args.E2FGVI_checkpoint, args.device)
    logger.setLevel(int(args.logger_level) or 20)
    frame_range = int(args.start_frame), int(args.end_frame)
    np_frames = executor.load_images(args.image_path, mode='RGB', frame_range=frame_range)
    np_masks = executor.load_images(args.mask_path, mode='P', frame_range=frame_range)

    executor.execute_inpainting(np_frames, np_masks, args.output, args.ratio)
