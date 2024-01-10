#!/usr/bin/env python3

import cv2

import numpy as np
# import svgwrite
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from magicroto.utils.logger import logger
from magicroto.utils.image_utils import get_mask_label_grid
from torch.nn import functional as F

# from mask_painter import mask_painter


class BaseSegmenter:
    def __init__(self, SAM_checkpoint, model_type, device='cuda:0'):
        """
        device: model device
        SAM_checkpoint: path of SAM checkpoint
        model_type: vit_b, vit_l, vit_h
        """
        logger.info(f"Initializing BaseSegmenter to {device}")
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], 'model_type must be vit_b, vit_l, or vit_h'

        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model = sam_model_registry[model_type](checkpoint=SAM_checkpoint)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.embedded = False
        logger.info(f"Successfully Initialize BaseSegmenter to {device}")

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.orignal_image = image
        if self.embedded:
            logger.info('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(image)
        self.embedded = True
        return

    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        self.predictor.reset_image()
        self.embedded = False


    @torch.no_grad()
    def resize_mask(self, mask):
        """
        Resize the mask tensor to match the shape of logits used in SAM model.

        @param mask: Input mask tensor.
        @return: Resized mask tensor.
        """
        # Visualize original mask
        # visualize_mask(mask, 'original_mask.png')
        # Convert to a PyTorch tensor if it's a NumPy array
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Ensure mask has a channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Adds channel dimension

        # Resize mask to 256x256
        resized_mask = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

        # Visualize resized mask
        # visualize_mask(resized_mask, 'resized_mask.png')

        return resized_mask

    @torch.no_grad()
    def sam_refinement(self, frame, logits, mode='point', num_points=5, erosion_size=3, random_points=True):
        """
        refine segmentation results with mask prompt
        """
        # convert to 1, 256, 256
        self.set_image(frame)
        # mode = 'mask'
        # Convert frame to tensor and get its size

        # frame = Image.open(frame).convert('RGB')
        if mode == 'mask':
            prompts = {'mask_input': self.resize_mask(logits)}  # 1 256 256
        else:
            points, labels = get_mask_label_grid(logits,
                                                 num_points=num_points, erosion_size=erosion_size, random_points=random_points)
            prompts = {
                'point_coords': np.array(points),
                'point_labels': np.array(labels),
            }
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        masks, scores, logits = self.predict(prompts, mode, multimask=True)

        self.reset_image()
        return masks.astype(np.uint8)


    def predict(self, prompts, mode, multimask=True):
        """
        image: numpy array, h, w, 3
        prompts: dictionary, 3 keys: 'point_coords', 'point_labels', 'mask_input'
        prompts['point_coords']: numpy array [N,2]
        prompts['point_labels']: numpy array [1,N]
        prompts['mask_input']: numpy array [1,256,256]
        mode: 'point' (points only), 'mask' (mask only), 'both' (consider both)
        mask_outputs: True (return 3 masks), False (return 1 mask only)
        whem mask_outputs=True, mask_input=logits[np.argmax(scores), :, :][None, :, :]
        """
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both'], 'mode must be point, mask, or both'

        if mode == 'point':
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'],
                                                           point_labels=prompts['point_labels'],
                                                           multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.predictor.predict(mask_input=prompts['mask_input'],
                                                           multimask_output=multimask)
        elif mode == 'both':  # both
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'],
                                                           point_labels=prompts['point_labels'],
                                                           mask_input=prompts['mask_input'],
                                                           multimask_output=multimask)
        else:
            raise ("Not implement yet!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        print("Shape of masks:", masks.shape)
        print("Shape of scores:", scores.shape)
        print("Shape of logits:", logits.shape)
        return masks, scores, logits


# def create_svg_from_array(data, filename):
#     # Find contours using OpenCV
#     contours, _ = cv2.findContours(data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     dwg = svgwrite.Drawing(filename, profile='tiny')

#     for contour in contours:
#         # Convert contour coordinates to SVG path
#         path_data = 'M ' + ' L '.join(f'{pt[0][0]},{pt[0][1]}' for pt in contour)
#         dwg.add(dwg.path(d=path_data, fill='black', stroke='black'))

#     dwg.save()


if __name__ == "__main__":
    # load and show an image
    iTest = r"D:/track_anything_project/Track-Anything/test_sample/test_01.jpg"
    image = cv2.imread(iTest)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # numpy array (h, w, 3)

    # initialise BaseSegmenter
    SAM_checkpoint = r'D:/track_anything_project/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    device = "cuda:0"
    base_segmenter = BaseSegmenter(SAM_checkpoint=SAM_checkpoint, model_type=model_type, device=device)

    # image embedding (once embedded, multiple prompts can be applied)
    base_segmenter.set_image(image)

    # examples
    # point only ------------------------
    mode = 'point'
    prompts = {
        'point_coords': np.array([[260, 525], [332, 672]]),
        'point_labels': np.array([1, 2]),
    }
    masks, scores, logits = base_segmenter.predict(prompts, mode,
                                                   multimask=False)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    print(type(masks))
    print(type(scores), scores)
    print(type(logits), logits)
    for i, m in enumerate(masks):
        im = (m * 255).astype(np.uint8)
        im = Image.fromarray(im, 'L')  # 'L' for grayscale
        im.save(f'mask{i}.png')

        # create_svg_from_array(m, f'mask{i}.svg')
    #
    # painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    # painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    # cv2.imwrite(r'D:\track_anything_project\Track-Anything\test_sample\truck_point.jpg', painted_image)

    # # both ------------------------
    # mode = 'both'
    # mask_input  = logits[np.argmax(scores), :, :]
    # prompts = {'mask_input': mask_input [None, :, :]}
    # prompts = {
    #     'point_coords': np.array([[500, 375], [1125, 625]]),
    #     'point_labels': np.array([1, 0]),
    #     'mask_input': mask_input[None, :, :]
    # }
    # masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    # painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    # painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    # cv2.imwrite('/hhd3/gaoshang/truck_both.jpg', painted_image)
    #
    # # mask only ------------------------
    # mode = 'mask'
    # mask_input  = logits[np.argmax(scores), :, :]
    #
    # prompts = {'mask_input': mask_input[None, :, :]}
    #
    # masks, scores, logits = base_segmenter.predict(prompts, mode, multimask=True)  # masks (n, h, w), scores (n,), logits (n, 256, 256)
    # painted_image = mask_painter(image, masks[np.argmax(scores)].astype('uint8'), background_alpha=0.8)
    # painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    # cv2.imwrite('/hhd3/gaoshang/truck_mask.jpg', painted_image)
