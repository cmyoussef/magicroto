"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from magicroto.core.rvm_pkg.model import MattingNetwork
from magicroto.core.rvm_pkg.inference_utils import ImageSequenceReader


def convert_video(model,
                  input_path: str,
                  ouput_path: str,
                  frame_range: tuple,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'png',
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    """
    Args:
        input_files:list of image paths.
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["png"].
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """

    assert output_type in ['png'], ' "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'


    # Initialize transform
    transform = transforms.ToTensor()
    frame_range = frame_range[0], frame_range[1]+1
    # Initialize reader
    source = ImageSequenceReader(input_path, frame_range, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)

    # Initialize writers
    frame_range = range(*frame_range)

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    # try:
    with torch.no_grad():
        bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
        rec = [None] * 4
        for i, src in enumerate(reader):
            src = src.to(device, dtype, non_blocking=True).unsqueeze(0)  # [B, T, C, H, W]
            fgr, pha, *rec = model(src, *rec, downsample_ratio)

            # for for t in range(frames.shape[0]):
            output_path = ouput_path.replace('.%04d.', f'.{frame_range[i]:04d}.')
            to_pil_image(pha[0][0]).save(output_path)
            bar.update(src.size(1))


class Converter:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device

    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)


if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-type', type=str, required=True, choices=['png'])
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    args = parser.parse_args()

    converter = Converter(args.variant, args.checkpoint, args.device)
    converter.convert(
        input_source=args.input_source,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress
    )
