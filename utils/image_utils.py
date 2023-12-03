import math
import os
import random

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
from PIL import ImageFilter
from PySide2.QtCore import QPoint
from PySide2.QtGui import QImage, QPainter, QColor, QPolygon, QPen

from magicroto.utils.logger import logger

roto_colors = [
    (128, 0, 128),  # Purple
    (0, 128, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 0, 0),  # Red
    (255, 255, 0),  # Yellow
    (0, 255, 255)  # Cyan
]


def convert_qimage_to_pil_image(qimage: QImage) -> Image.Image:
    # Convert QImage to bytes
    buffer = qimage.bits().tobytes()
    image_size = (qimage.width(), qimage.height())

    # Create a PIL Image using the QImage data
    if qimage.format() == QImage.Format.Format_RGB32:
        pil_image = Image.frombuffer('RGBA', image_size, buffer, 'raw', 'BGRA', 0, 1)
    elif qimage.format() == QImage.Format.Format_RGB888:
        pil_image = Image.frombuffer('RGB', image_size, buffer, 'raw', 'RGB', 0, 1)
    else:
        raise ValueError("Unsupported QImage format")

    return pil_image


def convert_pil_to_qimage(pil_im: Image.Image) -> QImage:
    # Get image data
    np_image = np.array(pil_im)

    # Convert NumPy array to QImage
    height, width, channel = np_image.shape
    bytes_per_line = channel * width
    if channel == 4:
        qim = QImage(np_image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    elif channel == 3:
        qim = QImage(np_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    else:
        raise ValueError("Unsupported channel count")

    return qim


def get_aspect_ratio(img):
    info = img.info
    if "aspect" in info:
        aspect_ratio = info['aspect']
    else:
        aspect_ratio = 1  # Default is square pixels

    return aspect_ratio


def get_contours(array, retrieval_mode=cv2.RETR_EXTERNAL, approximation=cv2.CHAIN_APPROX_NONE):
    return cv2.findContours(array.astype(np.uint8), retrieval_mode, approximation)


def fill_holes_in_boolean_array(boolean_array, kernel_size=5):
    # Convert the boolean array to uint8
    img = np.uint8(boolean_array) * 255

    # Create a kernel for the morphological operation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform the close operation
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Convert back to boolean array and return
    return closing == 255


def overlay_boolean_arrays_on_qimage(base_image: QImage, boolean_arrays, draw_contours=True):
    image_copy = QImage(base_image)

    painter = QPainter(image_copy)
    painter.setOpacity(0.7)  # Sets the transparency level
    color = QColor(128, 0, 128, 127)  # RGBA: Semi-transparent purple
    painter.setPen(color)

    drawn_pixels = set()

    for i, boolean_array in enumerate(boolean_arrays):
        boolean_array = fill_holes_in_boolean_array(boolean_array)

        # color = random.choice(roto_colors)
        color = roto_colors[i % len(roto_colors)]
        color = QColor(*color)  # RGBA: Semi-transparent purple
        painter.setPen(color)
        for i in range(len(boolean_array)):
            for j in range(len(boolean_array[i])):
                if boolean_array[i][j]:
                    coord = (j, i)
                    if coord not in drawn_pixels:
                        painter.drawPoint(j, i)  # Swap i and j
                        drawn_pixels.add(coord)

    # Draw contours with 1 opacity
    if draw_contours:
        painter.setOpacity(1.0)  # Full opacity for contours
        thick_pen = QPen(QColor(225, 225, 225, 255))  # RGBA: Fully opaque purple
        thick_pen.setWidth(3)  # Set line width to 3
        painter.setPen(thick_pen)
        boolean_array = np.logical_or.reduce(boolean_arrays, axis=0)
        boolean_array = fill_holes_in_boolean_array(boolean_array)
        # contours, _ = cv2.findContours(boolean_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = get_contours(boolean_array)
        for contour in contours:
            points = [QPoint(pt[0][0], pt[0][1]) for pt in contour]
            qpoly = QPolygon(points)
            painter.drawPolyline(qpoly)

    painter.end()
    return image_copy


def create_image_rgb(array_list):
    if len(array_list) == 3:
        # Convert each boolean array to an image channel
        channels = [Image.fromarray((255 * np.clip(arr, 0, 1)).astype('uint8')) for arr in array_list]

        # Combine the boolean arrays and create the alpha channel
        combined_alpha = np.logical_or.reduce(array_list)
        combined_alpha = fill_holes_in_boolean_array(combined_alpha)

        # Check if the combined_alpha contains any True values
        if not np.any(combined_alpha):
            print("Warning: The alpha channel is completely transparent.")

        alpha_channel = Image.fromarray((255 * np.clip(combined_alpha, 0, 1)).astype('uint8'))

        # Merge the RGB channels and the alpha channel into an RGBA image
        img_rgba = Image.merge("RGBA", channels + [alpha_channel])

        return img_rgba

    elif len(array_list) == 1:
        # Create a grayscale image for a single array
        arr = fill_holes_in_boolean_array(array_list[0])
        return Image.fromarray((255 * np.clip(arr, 0, 1)).astype('uint8'), 'L')

    else:
        raise ValueError("Array list must contain 1 or 3 arrays.")



def create_image(bool_array, color):
    bool_array = fill_holes_in_boolean_array(bool_array)
    rows, cols = len(bool_array), len(bool_array[0])
    # Initialize an empty image with alpha channel set to 0 (fully transparent)
    img = Image.new('RGBA', (cols, rows), (0, 0, 0, 0))
    pixels = img.load()

    for row in range(rows):
        for col in range(cols):
            if bool_array[row][col]:
                # Set the pixel to the given color with alpha set to 1 (fully opaque)
                pixels[col, row] = (*color, 255)

    return img


def overlay_boolean_arrays_on_pil(base_image: Image, boolean_arrays, draw_contours=True):
    image = base_image.convert('RGBA')
    drawn_pixels = set()

    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, boolean_array in enumerate(boolean_arrays):
        boolean_array = fill_holes_in_boolean_array(boolean_array)  # Assuming you have this function

        color = roto_colors[i % len(roto_colors)]
        for row in range(len(boolean_array)):
            for col in range(len(boolean_array[row])):
                if boolean_array[row][col]:
                    coord = (col, row)
                    if coord not in drawn_pixels:
                        draw.point((col, row), fill=color)
                        drawn_pixels.add(coord)

    if draw_contours:
        boolean_array = np.logical_or.reduce(boolean_arrays, axis=0)
        contours = get_contours(np.array(boolean_array))
        for contour in contours:
            rounded_contour = [(int(round(p[1])), int(round(p[0]))) for p in contour]
            draw.line(rounded_contour + [rounded_contour[0]], width=3, fill=(255, 255, 255, 255))

    image = Image.alpha_composite(image, overlay)

    return image


def overlay_boolean_arrays_on_image(image, boolean_arrays: list):
    # Convert PIL image to RGBA (if it's not)

    if isinstance(image, QImage):
        return overlay_boolean_arrays_on_qimage(image, boolean_arrays)
    elif isinstance(image, Image):
        return overlay_boolean_arrays_on_pil(image, boolean_arrays)
    else:
        raise ValueError(f"Unsupported image {type(image)}")


def load_image(img):
    if isinstance(img, list):
        img = img[0]

    if not os.path.isfile(img):
        logger.error(f'File does not exist {img}')
        return None

    img = Image.open(img.replace('/', '/'))
    aspect_ratio = get_aspect_ratio(img)
    if aspect_ratio != 1:
        logger.warning(f"Stable-diffusion works with pixel aspect 1. Your image aspect is {aspect_ratio}")

    # Resizing the image to be divisible by 8 if needed
    img = module_8_resize(img)

    return img


def module_8_resize(img):
    width, height = img.size

    new_width = ((width // 8) + (width % 8 > 0)) * 8
    new_height = ((height // 8) + (height % 8 > 0)) * 8

    if new_width != width or new_height != height:
        img = img.resize((new_width, new_height), Image.LANCZOS)
        logger.warning("Image resized to {}x{} for compatibility with stable-diffusion.".format(new_width, new_height))

    return img


def randomly_modify_image(image):
    # to test the split blend
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.5, 1.5)
    return enhancer.enhance(factor)


class ImageSplitMerge:

    def __init__(self, overlap_percent=0, gaussian=True):
        self.original_size = None
        self.chunk_width = None
        self.chunk_height = None
        self.overlap_x = 0
        self.overlap_y = 0
        self.gaussian = gaussian
        self.overlap_percent = overlap_percent / 100.0  # Convert to a fraction
        self.chunk_border_list = []

    def create_feathered_mask(self, borders=(True, True, True, True)):
        left_border, upper_border, right_border, lower_border = borders

        x_shift = self.overlap_x if left_border else 0
        y_shift = self.overlap_y if upper_border else 0

        feather_width = self.chunk_width + x_shift + (self.overlap_x if right_border else 0)
        feather_height = self.chunk_height + y_shift + (self.overlap_y if lower_border else 0)

        # print(feather_width / self.chunk_width, feather_height/ self.chunk_height)
        mask = Image.new("L", (feather_width, feather_height), color=0)
        draw = ImageDraw.Draw(mask)

        draw.rectangle([(x_shift, y_shift),
                        (self.chunk_width + x_shift, self.chunk_height + y_shift)],
                       fill="white", outline="white")
        # return mask

        # Loop as spiral to create the feathering effect
        for layer in range(max(self.overlap_x, self.overlap_y)):

            if self.gaussian:
                alpha = 255 - int(255 * math.exp(-(layer ** 2) / (2 * (self.overlap_x / 3) ** 2)))
            else:
                alpha = int(255 * (layer / self.overlap_x))

            if upper_border:
                start = layer if left_border else 0
                end = layer if right_border else 0
                for x in range(start, feather_width - end):
                    draw.point((x, layer), fill=alpha)

            if lower_border:
                start = layer if left_border else 0
                end = layer if right_border else 0
                for x in range(start, feather_width - end):
                    draw.point((x, feather_height - 1 - layer), fill=alpha)

            if left_border:
                start = layer if upper_border else 0
                end = layer if lower_border else 0
                for y in range(start, feather_height - end):
                    draw.point((layer, y), fill=alpha)
            if right_border:
                start = layer if upper_border else 0
                end = layer if lower_border else 0
                for y in range(start, feather_height - end):
                    draw.point((feather_width - 1 - layer, y), fill=alpha)

        return mask

    def unpremultiply(self, image):
        img_array = np.array(image)

        if img_array.shape[2] < 4:
            return image  # No alpha channel, so no need to unpremultiply

        alpha = img_array[:, :, 3] / 255.0
        img_array[:, :, :3] = img_array[:, :, :3] / (
                alpha[:, :, None] + 1e-7)  # Added small epsilon to avoid divide by zero
        img_array[alpha == 0] = 0

        return Image.fromarray(img_array.astype('uint8'), 'RGBA')

    def set_alpha_channel_to_one(self, img):
        img_array = np.array(img)
        img_array[..., 3] = 255  # set alpha channel to 1
        img_with_alpha_one = Image.fromarray(img_array, 'RGBA')
        return img_with_alpha_one

    def split_image(self, image, split_size):

        self.original_size = image.size  # Store original size
        width, height = image.size
        # Calculate the size of each chunk
        num_splits_x = int(np.ceil(width / split_size))
        num_splits_y = int(np.ceil(height / split_size))

        self.chunk_width = width // num_splits_x
        self.chunk_height = height // num_splits_y

        self.overlap_x = int(self.chunk_width * self.overlap_percent * .5)
        self.overlap_y = int(self.chunk_height * self.overlap_percent * .5)

        # To make self.overlap_x and self.overlap_y divisible by 8,
        self.overlap_x = (round(self.overlap_x / 8)) * 8
        self.overlap_y = (round(self.overlap_y / 8)) * 8

        chunks = []

        def get_border_value(val, limit):
            new_val = min(max([0, val]), limit)
            overlap = val == new_val
            return new_val, overlap

        for i in range(0, height, self.chunk_height):
            for j in range(0, width, self.chunk_width):
                left, left_border = get_border_value(j - self.overlap_x, width)
                upper, upper_border = get_border_value(i - self.overlap_y, height)
                right, right_border = get_border_value(j + self.chunk_width + self.overlap_x, width)
                lower, lower_border = get_border_value(i + self.chunk_height + self.overlap_y, height)

                self.chunk_border_list.append((left_border, upper_border, right_border, lower_border))
                chunk = image.crop((left, upper, right, lower))
                chunks.append(chunk)
                # chunk.show()
        return chunks

    def update_scaling_ratios(self, ratio_x, ratio_y):
        self.original_size = int(self.original_size[0] * ratio_x), int(self.original_size[1] * ratio_y)
        self.chunk_width = int(self.chunk_width * ratio_x)
        self.chunk_height = int(self.chunk_height * ratio_y)
        self.overlap_x = int(self.overlap_x * ratio_x)
        self.overlap_y = int(self.overlap_y * ratio_y)

    def merge_images(self, chunks):
        # Assuming the first chunk represents the scaling ratios for all
        ratio_x = int(chunks[0].width / self.chunk_width)
        ratio_y = int(chunks[0].height / self.chunk_height)

        self.update_scaling_ratios(ratio_x, ratio_y)

        width, height = self.original_size
        new_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        idx = 0
        for i in range(0, height, self.chunk_height):
            for j in range(0, width, self.chunk_width):
                # Calculate actual dimensions for this specific chunk
                borders = self.chunk_border_list[idx]
                feathered_mask = self.create_feathered_mask(
                    borders)  # .resize(chunks[idx].size, Image.Resampling.BICUBIC)

                # Get the chunk and modify it

                left_border, upper_border, right_border, lower_border = borders
                x_shift = (self.overlap_x if left_border else 0)  # + (self.overlap_x if right_border else 0)
                y_shift = (self.overlap_y if upper_border else 0)  # + (self.overlap_y if lower_border else 0)

                # chunks[idx] = randomly_modify_image(chunks[idx])
                new_image.paste(chunks[idx], (j - x_shift, i - y_shift), mask=feathered_mask)
                # print(idx)
                idx += 1

        new_image = self.unpremultiply(new_image)
        new_image = self.set_alpha_channel_to_one(new_image)
        new_image = new_image.convert('RGB')
        return new_image


class ImageMaskCropper:
    def __init__(self, image, mask, buffer_percent=10):
        self.image = image
        mask_img = mask

        # Check for alpha channel
        if mask_img.mode == 'RGBA':
            self.mask = mask_img.split()[-1]
        else:
            self.mask = mask_img.convert('L')

        self.bbox = None
        self.bbox_with_buffer = None
        self.buffer_percent = buffer_percent / 100.0  # Convert percentage to a ratio

        # Match the mask size to the image size
        self.mask = self.mask.resize(self.image.size, Image.LANCZOS)

    def find_bounding_box(self):
        mask_array = np.array(self.mask, dtype=np.uint8)
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        self.bbox = (xmin, ymin, xmax, ymax)

        # Apply buffer
        x_buffer = int((xmax - xmin) * self.buffer_percent)
        y_buffer = int((ymax - ymin) * self.buffer_percent)

        width, height = self.image.size
        xmin = max(0, xmin - x_buffer)
        ymin = max(0, ymin - y_buffer)
        xmax = min(width, xmax + x_buffer)
        ymax = min(height, ymax + y_buffer)

        self.bbox_with_buffer = (xmin, ymin, xmax, ymax)

    def create_feathered_mask(self):
        # Initial setup for bounding boxes
        x_min, y_min, x_max, y_max = self.bbox_with_buffer
        width, height = x_max - x_min, y_max - y_min

        # Create a black image
        feathered_mask = Image.new("L", (width, height), color=0)

        # Calculate the position for the white rectangle
        inner_x_min = self.bbox[0] - x_min
        inner_y_min = self.bbox[1] - y_min
        inner_x_max = self.bbox[2] - x_min
        inner_y_max = self.bbox[3] - y_min

        # Create a PIL ImageDraw object and draw the white rectangle

        draw = ImageDraw.Draw(feathered_mask)
        draw.rectangle([inner_x_min, inner_y_min, inner_x_max, inner_y_max], fill=255)

        def draw_border(size):
            draw.rectangle([0, 0, width - 1, size], fill=0)  # Top
            draw.rectangle([0, height - size, width - 1, height - 1], fill=0)  # Bottom
            draw.rectangle([0, 0, size, height - 1], fill=0)  # Left
            draw.rectangle([width - size, 0, width - 1, height - 1], fill=0)

        # Perform additive blending of the white rectangle with blur
        iterations = int(min([inner_y_min // 2, inner_x_min // 2]) * .7)
        for _ in range(iterations):
            draw_border(5)
            feathered_mask = feathered_mask.filter(ImageFilter.GaussianBlur(5))
            draw.rectangle([inner_x_min, inner_y_min, inner_x_max, inner_y_max], fill=255)
            draw_border(5)

        self.feathered_mask = feathered_mask  # Image.fromarray(feathered_mask, 'L')
        return self.feathered_mask

    def crop_image(self):
        if self.bbox_with_buffer is None:
            self.find_bounding_box()

        self.feathered_mask = self.create_feathered_mask()
        self.cropped_image = self.image.crop(self.bbox_with_buffer)
        self.cropped_mask = self.mask.crop(self.bbox_with_buffer)
        return self.cropped_image, self.cropped_mask

    def merge_back(self, modified_cropped_image):
        modified_cropped_image = modified_cropped_image.resize(self.cropped_image.size, Image.LANCZOS)
        self.image.paste(modified_cropped_image, (self.bbox_with_buffer[0], self.bbox_with_buffer[1]),
                         mask=self.feathered_mask)
        return self.image


if __name__ == "__main__":
    img = r"C:\Users\mellithy\nuke-stable-diffusion\sd_inPaint\cyberrealistic_v31\source_img2img/init_img.png"
    msk = r"C:\Users\mellithy\nuke-stable-diffusion\sd_inPaint\cyberrealistic_v31\source_img2img/mask_img.png"
    # img = Image.open(img)
    img = Image.open(img)
    msk = Image.open(msk)
    img = img.resize((1024, 512), Image.Resampling.NEAREST)
    img_cropper = ImageMaskCropper(img, msk)
    crop_img = img_cropper.crop_image()
    # crop_img.show()
    crop_img_edited = randomly_modify_image(crop_img)
    merge_back = img_cropper.merge_back(crop_img_edited)
    merge_back.show()
    mask = img_cropper.create_feathered_mask()
    mask.show()
    # Usage
    # splitter = ImageSplitMerge(overlap_percent=0)
    # image = Image.open(
    #     "C:/Users/mellithy/nuke-stable-diffusion/SD_Img2Img/models--stabilityai--stable-diffusion-xl-refiner-1.0/20230921_072109/20230921_072109_1_1.png")
    # chunk_width = 512  # Set this based on your needs
    #
    # chunks = splitter.split_image(image, chunk_width)
    # for i, c in enumerate(chunks):
    #     chunks[i] = c.resize([c.width * 2, c.height * 2], Image.LANCZOS)
    # image.show()
    # mask = splitter.create_feathered_mask()
    # mask.show()
    # merged_image = splitter.merge_images(chunks)
    # merged_image.show()
