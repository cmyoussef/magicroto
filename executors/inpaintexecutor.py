import argparse
import inspect
import os
import sys

import torch
from PIL import Image

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
        sys.path.append(p)

# </editor-fold>

from magicroto.core.inpainter.base_inpainter import BaseInpainter
from magicroto.utils.logger import logger

from magicroto.utils import image_utils


class InPaintExecutor:
    def __init__(self, args):
        """
        Initialize the InPaintExecutor with a BaseInpainter instance.

        @param E2FGVI_checkpoint: Path to the model checkpoint.
        @param device: Device to use for computation.
        """
        self.args = args
        E2FGVI_checkpoint = self.args.get('E2FGVI_checkpoint')
        device = self.args.get('device')

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
        parser.add_argument('--chunk_size', type=int, default=0, help='To split the inpaint into chunks')
        return parser

    def execute_inpainting(self, frames, masks, output, ratio=1, start_frame=0, chunk_size=0):
        """
        Execute inpainting on provided frames and masks.

        @param frames: Numpy array of frames.
        @param masks: Numpy array of masks.
        @param output: Directory to save inPainted images.
        @param ratio: Down-sampling ratio.
        @param start_frame: to name the files correctly.
        @param chunk_size: to split the inpaint into chunks.
        @return: None
        """
        if len(frames) != len(masks):
            raise ValueError("The number of frames and masks must be the same.")

        def process_chunk(frame_chunk, mask_chunk, start_idx):
            inPainted_frames_chunk = self.base_inPaint.inpaint(frame_chunk, mask_chunk, ratio=ratio)
            for ti, inPainted_frame in enumerate(inPainted_frames_chunk):
                frame = Image.fromarray(inPainted_frame).convert('RGB')
                frame.save(output.replace('.%04d.', f'.{ti + start_idx:04d}.'))
            torch.cuda.empty_cache()

        num_frames = len(frames)
        if chunk_size <= 0 or chunk_size >= num_frames:
            process_chunk(frames, masks, start_frame)
        else:
            for i in range(0, num_frames, chunk_size):
                end_idx = min(i + chunk_size, num_frames)
                frame_chunk = frames[i:end_idx]
                mask_chunk = masks[i:end_idx]
                process_chunk(frame_chunk, mask_chunk, start_frame + i)


if __name__ == '__main__':
    args = InPaintExecutor.setup_parser().parse_args()
    executor = InPaintExecutor(vars(args))
    logger.setLevel(int(args.logger_level) or 20)
    frame_range = int(args.start_frame), int(args.end_frame)
    np_frames = image_utils.load_images(args.image_path, mode='RGB', frame_range=frame_range)
    np_masks = image_utils.load_images(args.mask_path, mode='P', frame_range=frame_range)

    executor.execute_inpainting(np_frames, np_masks, args.output, args.ratio,
                                start_frame=args.start_frame, chunk_size=args.chunk_size)
