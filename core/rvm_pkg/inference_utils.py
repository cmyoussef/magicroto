import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image


class ImageSequenceReader(Dataset):
    def __init__(self, path, frame_range, transform=None):
        # super().__init__()
        self.transform = transform
        self.path = path
        self.frame_range = range(*frame_range)

    def get_file(self, frame):
        return self.path.replace('.%04d.', f'.{self.frame_range[frame]:04d}.')

    def __len__(self):
        return len(self.frame_range)

    def __getitem__(self, idx):
        with Image.open(self.get_file(idx)) as img:
            img.load()
        if self.transform is not None:
            return self.transform(img)
        return img


class ImageSequenceWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)

    def write(self, frames):
        # frames: [T, C, H, W]
        for t in range(frames.shape[0]):
            to_pil_image(frames[t]).save(os.path.join(
                self.path, str(self.counter).zfill(4) + '.' + self.extension))
            self.counter += 1
