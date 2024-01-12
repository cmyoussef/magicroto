import argparse
import json
import os
import sys

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

from magicroto.core.tracker.base_tracker import BaseTracker
from magicroto.core.base_segmenter import BaseSegmenter

from magicroto.utils.logger import logger
from magicroto.utils import image_utils
from magicroto.core.tracker.util import painter


class TrackingExecutor:
    def __init__(self, XMEM_checkpoint, args):
        """
        Initialize the TrackingExecutor with a BaseTracker instance.

        @param XMEM_checkpoint: Path to the model checkpoint.
        """
        self.args = vars(args)
        self.track_checkpoint = XMEM_checkpoint

        self.refine_mode = self.args.get('refine_mode', 1)
        self.sam_model = None
        SAM_checkpoint = self.args.get('SAM_checkpoint', None)
        model_type = 'vit_h'
        device = self.args.get('device', 'cuda:0')
        if SAM_checkpoint and os.path.exists(SAM_checkpoint):
            self.sam_model = BaseSegmenter(SAM_checkpoint=SAM_checkpoint, model_type=model_type, device=device)

    @staticmethod
    def setup_parser():
        """
        Set up command-line argument parsing.
        @return: ArgumentParser object.
        """
        parser = argparse.ArgumentParser(description='tracking images using BaseTracking.')
        parser.add_argument('--cache_dir', required=True, help='Path to the model checkpoint')
        parser.add_argument('--XMEM_checkpoint', required=True, help='Path to the model checkpoint')
        parser.add_argument("--SAM_checkpoint", type=str, required=False, help="Path to the SAM checkpoint.")
        parser.add_argument("--refine_mode", type=str, required=False, help="To use SAM refiner.")

        parser.add_argument('--device', default='cuda:0', help='Device to use for computation')
        parser.add_argument('--image_path', required=True, help='Wildcard path to the frame images')
        parser.add_argument('--mask_path', required=True, help='Wildcard path to the mask images')
        parser.add_argument('--output', required=True, help='Directory to save tracked mask images')
        parser.add_argument('--ratio', type=float, default=1, help='Down-sampling ratio')
        parser.add_argument('--logger_level', type=int, default=1001, help='First frame to paint')
        parser.add_argument('--mask_frames', type=json.loads, default=1001, help='List of frame numbers for the mask')
        parser.add_argument('--start_frame', type=int, default=1001, help='First frame to paint')
        parser.add_argument('--end_frame', type=int, default=1001, help='Last frame to paint')
        return parser

    @staticmethod
    def track_masks(start_frame, end_frame, mask_frames):
        """
        Track masks over a specified range of frames.

        @param start_frame: Starting frame of the range.
        @param end_frame: Ending frame of the range.
        @param mask_frames: List of frames where masks are manually created.
        @return: List of tuples with mask frame and frame range to track.
        """
        result = []

        # Handle the case where the start frame is before the first mask frame
        if start_frame < mask_frames[0]:
            result.append((mask_frames[0], (mask_frames[0], start_frame - 1)))

        # Iterate through the mask frames
        for i in range(len(mask_frames)):
            # For the last mask frame, track until the end frame
            if i == len(mask_frames) - 1:
                range_end = end_frame + 1
            else:
                # For other mask frames, track until the frame before the next mask frame
                range_end = mask_frames[i + 1]

            result.append((mask_frames[i], (mask_frames[i], range_end)))

        return result

    def prepr_args(self):
        self.args['frame_range'] = int(self.args['start_frame']), int(self.args['end_frame'])

        self.args['frames_to_track'] = TrackingExecutor.track_masks(self.args['start_frame'], self.args['end_frame'],
                                                                    self.args['mask_frames'])
        logger.debug(f"frames_to_track: self.args['frames_to_track']")

        self.args['np_frames'] = image_utils.load_images(self.args['image_path'], mode='RGB',
                                                         frame_range=self.args['frame_range'], return_dict=True)
        logger.info(f'frames loaded: {len(self.args["np_frames"])} frames')

        self.args['np_masks'] = image_utils.load_images(self.args['mask_path'], mode='P',
                                                        frames=self.args['mask_frames'], return_dict=True)
        logger.info(f'frames loaded: {len(self.args["np_masks"])} masks')

    def execute_tracking(self):
        """
        Execute inpainting on provided frames and masks.

        @param frames: Numpy array of frames.
        @param masks: Numpy array of masks.
        @param output: Directory to save inPainted images.
        @return: None
        """
        exhaustive = False if len(self.args['frames_to_track']) == 1 else False
        # inPainted_frames = self.base_track.tracking(frames, masks, ratio=ratio)
        for i, (mask, (start, end)) in logger.progress(enumerate(self.args['frames_to_track'])):
            # Here, 'mask' is your mask frame, and 'start' and 'end' are the frame ranges
            np_mask = self.args['np_masks'].get(mask)
            np_frame = self.args['np_frames'].get(start)
            self.tracker = BaseTracker(self.track_checkpoint)
            logger.info(
                f"Processing mask at frame {mask:04d} exhaustive:{exhaustive}, refiner_mode:{self.refine_mode}, {np_mask.shape} from frame {start:04d} to {end:04d}")
            final_combined_mask, individual_masks = self.tracker.track(np_frame, np_mask, exhaustive=exhaustive)
            # if self.refine_mode is not None and self.sam_model:
            if self.sam_model:
                individual_masks = self.sam_model.sam_refinement(np_frame, final_combined_mask)
            img_path = self.args['output'].replace('.%04d.', f'.{start:04d}.')
            img = image_utils.create_image_rgb(final_combined_mask)
            img.save(img_path)
            # painter.create_exr_image(individual_masks,img_path)

            iterator = 1 if start < end else -1
            # Example processing: Loop through each frame in the range
            for it, frame in logger.progress(enumerate(range(start, end, iterator)), desc='Processing frames',
                                             total=abs(end - start)):  # +1 to include the end frame
                np_frame = self.args['np_frames'].get(frame)
                final_combined_mask, individual_masks = self.tracker.track(np_frame, exhaustive=exhaustive)
                # if self.refine_mode is not None and self.sam_model:
                if self.sam_model:
                    logger.debug(f"refining frame using {self.refine_mode}")
                    individual_masks = self.sam_model.sam_refinement(np_frame, final_combined_mask)

                img_path = self.args['output'].replace('.%04d.', f'.{frame:04d}.')
                img = image_utils.create_image_rgb(final_combined_mask)
                img.save(img_path)
                # painter.create_exr_image(individual_masks, img_path)
                # logger.debug(f"    Processing frame {frame}")

        self.tracker.clear_memory()


if __name__ == '__main__':
    args = TrackingExecutor.setup_parser().parse_args()
    logger.setLevel(int(args.logger_level) or 20)
    logger.debug(json.dumps(vars(args), indent=4))

    executor = TrackingExecutor(args.XMEM_checkpoint, args)
    logger.info('Tracker has been initialized')

    executor.prepr_args()
    executor.execute_tracking()
    # executor.execute_tracking(np_frames, np_masks, args.output, args.ratio)
