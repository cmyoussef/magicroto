import glob
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

# Custom imports
from inference.inference_core import InferenceCore
from tools.painter import mask_painter, create_exr_image
from tracker.model.network import XMem
from tracker.util.mask_mapper import MaskMapper
from tracker.util.range_transform import im_normalization


class BaseTracker:
    def __init__(self, xmem_checkpoint, device) -> None:
        """
        Initialize the BaseTracker.

        @param xmem_checkpoint: Checkpoint of XMem model.
        @param device: Model device.
        @return: None
        """
        # Load configurations
        dir_ = os.path.dirname(__file__)
        with open(os.path.join(dir_, 'config', 'config.yaml'), 'r') as stream:
            config = yaml.safe_load(stream)

        # Initialize XMem and InferenceCore
        network = XMem(config, xmem_checkpoint).to(device).eval()
        self.tracker = InferenceCore(network, config)

        # Data transformation
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        self.device = device

        # Changable properties
        self.mapper = MaskMapper()
        self.initialised = False

    @torch.no_grad()
    def resize_mask(self, mask):
        """
        Resize the mask.

        @param mask: Input mask to resize.
        @return: Resized mask.
        """
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h / min_hw * self.size), int(w / min_hw * self.size)),
                             mode='nearest')

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Track the given frame and return masks.

        @param frame: Input frame as a numpy array (H, W, 3).
        @param first_frame_annotation: First frame annotation (optional).
        @return: Tuple of final combined mask, individual masks.
        """
        mask, labels = (None, None)
        if first_frame_annotation is not None:
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))

        frame_tensor = self.im_transform(frame).to(self.device)
        probs, _ = self.tracker.step(frame_tensor, mask, labels)

        out_mask = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.uint8)
        final_combined_mask = np.zeros_like(out_mask)
        individual_masks = []

        # Map back and combine masks
        for k, v in self.mapper.remappings.items():
            object_mask = (out_mask == v).astype(np.uint8)
            final_combined_mask += object_mask * k
            individual_masks.append(object_mask)

        return final_combined_mask, individual_masks

    @staticmethod
    def paint_masks_on_image(individual_masks, image=None):
        """
        Paints each mask from individual masks onto an image or a black background if no image is provided.

        @param individual_masks: List of masks for each object.
        @param image: Optional; the image on which to paint the masks. If None, a black image is used.
        @return: Image with painted masks.
        """
        if image is None:
            # Assuming all masks have the same dimensions
            height, width = individual_masks[0].shape[:2]
            image = np.zeros((height, width, 3), dtype=np.uint8)

        painted_image = image.copy()
        for idx, mask in enumerate(individual_masks):
            mask_color = idx + 1  # Assuming each index corresponds to a unique color
            painted_image = mask_painter(painted_image, mask, mask_color, mask_alpha=0.7, contour_color=1,
                                         contour_width=3)

        return painted_image

    @torch.no_grad()
    def sam_refinement(self, frame, logits, ti):
        """
        refine segmentation results with mask prompt
        """
        # convert to 1, 256, 256
        self.sam_model.set_image(frame)
        mode = 'mask'
        logits = logits.unsqueeze(0)
        logits = self.resizer(logits).cpu().numpy()
        prompts = {'mask_input': logits}  # 1 256 256
        masks, scores, logits = self.sam_model.predict(prompts, mode,
                                                       multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
        painted_image = mask_painter(frame, masks[np.argmax(scores)].astype('uint8'), mask_alpha=0.8)
        painted_image = Image.fromarray(painted_image)
        painted_image.save(f'/ssd1/gaomingqi/refine/{ti:05d}.png')
        self.sam_model.reset_image()

    @torch.no_grad()
    def clear_memory(self):
        """
        Clear memory caches.

        @return: None
        """
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()


##  how to use:
##  1/3) prepare device and xmem_checkpoint
#   device = 'cuda:2'
#   XMEM_checkpoint = '/ssd1/gaomingqi/checkpoints/XMem-s012.pth'
##  2/3) initialise Base Tracker
#   tracker = BaseTracker(XMEM_checkpoint, device, None, device)    # leave an interface for sam model (currently set None)
##  3/3) 


if __name__ == '__main__':
    # video frames (take videos from DAVIS-2017 as examples)
    video_path_list = glob.glob(os.path.join(
        r'C:/Users/mellithy/magicRoto/MR_MagicRotoSelectorLive_avenger/source_MR_MagicRotoSelectorLive_avenger/',
        '*.png'))
    video_path_list.sort()
    print(video_path_list)
    # load frames
    frames = []
    for video_path in video_path_list:
        frames.append(Image.open(video_path).convert('RGB'))
    frames = np.stack(frames, 0)  # T, H, W, C
    # load first frame annotation
    first_frame_path = r'C:/Users/mellithy/magicRoto/MR_MagicRotoSelectorLive_avenger/20231213/mask.0001.png'
    first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))  # H, W, C

    # ------------------------------------------------------------------------------------
    # how to use
    # ------------------------------------------------------------------------------------
    # 1/4: set checkpoint and device
    device = 'cuda:0'
    XMEM_checkpoint = r'D:/track_anything_project/Track-Anything/checkpoints/XMem-s012.pth'
    # SAM_checkpoint= '/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth'
    # model_type = 'vit_h'
    # ------------------------------------------------------------------------------------
    # 2/4: initialise inpainter
    tracker = BaseTracker(XMEM_checkpoint, device)
    # ------------------------------------------------------------------------------------
    # 3/4: for each frame, get tracking results by tracker.track(frame, first_frame_annotation)
    # frame: numpy array (H, W, C), first_frame_annotation: numpy array (H, W), leave it blank when tracking begins
    individual_masks_list = []
    for ti, frame in enumerate(frames):
        if ti == 0:
            final_combined_mask, individual_masks = tracker.track(frame, first_frame_annotation)
            # mask: 
        else:
            final_combined_mask, individual_masks = tracker.track(frame)

        individual_masks_list.append(individual_masks)
    # ----------------------------------------------
    # 3/4: clear memory in XMEM for the next video
    tracker.clear_memory()
    # ----------------------------------------------
    # end
    # ----------------------------------------------
    print(f'max memory allocated: {torch.cuda.max_memory_allocated() / (2 ** 20)} MB')
    # set saving path
    save_path = 'C:/Users/mellithy/magicRoto/MR_MagicRotoSelectorLive_avenger/20231213/tracking_ouput/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # save
    for ti, im in enumerate(individual_masks_list):
        # painted_frame = tracker.paint_masks_on_image(im)
        create_exr_image(im, f'{save_path}/{ti:05d}')
        # painted_frame = Image.fromarray(painted_frame)
        # painted_frame.save(f'{save_path}/{ti:05d}.png')

    # tracker.clear_memory()
    # for ti, frame in enumerate(frames):
    #     print(ti)
    #     # if ti > 200:
    #     #     break
    #     if ti == 0:
    #         mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
    #     else:
    #         mask, prob, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/results/TrackA/gsw/{ti:05d}.png')

    # # track anything given in the first frame annotation
    # for ti, frame in enumerate(frames):
    #     if ti == 0:
    #         mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
    #     else:
    #         mask, prob, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/results/TrackA/horsejump-high/{ti:05d}.png')

    # # ----------------------------------------------------------
    # # another video
    # # ----------------------------------------------------------
    # # video frames
    # video_path_list = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/camel', '*.jpg'))
    # video_path_list.sort()
    # # first frame
    # first_frame_path = '/ssd1/gaomingqi/datasets/davis/Annotations/480p/camel/00000.png'
    # # load frames
    # frames = []
    # for video_path in video_path_list:
    #     frames.append(np.array(Image.open(video_path).convert('RGB')))
    # frames = np.stack(frames, 0)    # N, H, W, C
    # # load first frame annotation
    # first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C

    # print('first video done. clear.')

    # tracker.clear_memory()
    # # track anything given in the first frame annotation
    # for ti, frame in enumerate(frames):
    #     if ti == 0:
    #         mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
    #     else:
    #         mask, prob, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/results/TrackA/camel/{ti:05d}.png')

    # # failure case test
    # failure_path = '/ssd1/gaomingqi/failure'
    # frames = np.load(os.path.join(failure_path, 'video_frames.npy'))
    # # first_frame = np.array(Image.open(os.path.join(failure_path, 'template_frame.png')).convert('RGB'))
    # first_mask = np.array(Image.open(os.path.join(failure_path, 'template_mask.png')).convert('P'))
    # first_mask = np.clip(first_mask, 0, 1)

    # for ti, frame in enumerate(frames):
    #     if ti == 0:
    #         mask, probs, painted_image = tracker.track(frame, first_mask)
    #     else:
    #         mask, probs, painted_image = tracker.track(frame)
    #     # save
    #     painted_image = Image.fromarray(painted_image)
    #     painted_image.save(f'/ssd1/gaomingqi/failure/LJ/{ti:05d}.png')
    #     prob = Image.fromarray((probs[1].cpu().numpy()*255).astype('uint8'))

    #     # prob.save(f'/ssd1/gaomingqi/failure/probs/{ti:05d}.png')
