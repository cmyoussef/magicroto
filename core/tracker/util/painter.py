# paint masks, contours, or points on images, with specified colors
import Imath
import OpenEXR
import cv2
import numpy as np
from PIL import Image


def colormap(rgb=True):
    """
    Generates a colormap.

    @param rgb: Boolean indicating whether the colormap is in RGB format.
    @return: A list of colors.
    """
    color_list = np.array(
        [
            0.000, 0.000, 0.000,
            1.000, 1.000, 1.000,
            1.000, 0.498, 0.313,
            0.392, 0.581, 0.929,
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857
        ]
    ).astype(np.float32).reshape((-1, 3)) * 255
    return color_list if rgb else color_list[:, ::-1]


color_list = colormap().astype('uint8').tolist()


def vis_add_mask(image, mask, color, alpha):
    """
    Adds a colored mask to an image.

    @param image: Image to which the mask will be added.
    @param mask: Mask to be added to the image.
    @param color: Color of the mask.
    @param alpha: Transparency of the mask.
    @return: Image with mask added.
    """
    color = np.array(color_list[color])
    mask = mask > 0.5
    image[mask] = image[mask] * (1 - alpha) + color * alpha
    return image.astype('uint8')


def create_point_mask(shape, points, radius):
    """
    Creates a point mask.

    @param shape: Shape of the mask (height, width).
    @param points: List of points.
    @param radius: Radius of points.
    @return: Point mask.
    """
    h, w = shape
    point_mask = np.zeros((h, w), dtype='uint8')
    for point in points:
        cv2.circle(point_mask, (point[0], point[1]), radius, 1, -1)
    return point_mask


def create_contour_mask(mask, contour_width):
    """
    Creates a contour mask.

    @param mask: Binary mask.
    @param contour_width: Width of the contour.
    @return: Contour mask.
    """
    contour_radius = (contour_width - 1) // 2
    dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_transform_back = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    dist_map = dist_transform_fore - dist_transform_back
    contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
    contour_mask /= np.max(contour_mask)
    contour_mask[contour_mask > 0.5] = 1.
    return contour_mask


def point_painter(input_image, input_points, point_color=5, point_alpha=0.9, point_radius=15, contour_color=2,
                  contour_width=5):
    """
    Paints points and their contours on an image.

    @param input_image: Image to paint on.
    @param input_points: List of points to paint.
    @param point_color: Color of the points.
    @param point_alpha: Transparency of the points.
    @param point_radius: Radius of the points.
    @param contour_color: Color of the point contours.
    @param contour_width: Width of the point contours.
    @return: Image with points and contours painted.
    """
    assert input_image.shape[:2] == input_mask.shape, 'Image and mask shapes are different.'

    point_mask = create_point_mask(input_image.shape[:2], input_points, point_radius)
    contour_mask = create_contour_mask(point_mask, contour_width)

    painted_image = vis_add_mask(input_image.copy(), point_mask, point_color, point_alpha)
    painted_image = vis_add_mask(painted_image, 1 - contour_mask, contour_color, 1)
    return painted_image


def mask_painter(input_image, input_mask, mask_color=5, mask_alpha=0.7, contour_color=1, contour_width=3):
    """
    Paints a mask with optional contour on an image.

    @param input_image: Image to paint on.
    @param input_mask: Mask to be painted.
    @param mask_color: Color of the mask.
    @param mask_alpha: Transparency of the mask.
    @param contour_color: Color of the contour.
    @param contour_width: Width of the contour.
    @return: Image with mask and contour painted.
    """
    if input_image.shape[:2] != input_mask.shape:
        input_mask = cv2.resize(input_mask, (input_image.shape[1], input_image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    mask = np.clip(input_mask, 0, 1)
    contour_mask = create_contour_mask(mask, contour_width)

    painted_image = vis_add_mask(input_image.copy(), mask, mask_color, mask_alpha)
    painted_image = vis_add_mask(painted_image, 1 - contour_mask, contour_color, 1)
    return painted_image


def background_remover(input_image, input_mask):
    """
    Removes the background from an image using a mask.

    @param input_image: The image from which the background will be removed.
    @param input_mask: The mask used for background removal.
    @return: Image without the background (PIL.Image in RGBA format).
    """
    assert input_image.shape[:2] == input_mask.shape, 'Image and mask shapes are different.'

    # Apply mask to image
    mask = np.clip(input_mask, 0, 1).astype('uint8')
    rgba_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2BGRA)
    rgba_image[:, :, 3] = mask * 255

    return Image.fromarray(rgba_image, 'RGBA')


def convert_to_exr_format(image):
    """
    Convert an image to EXR format.

    @param image: Image to be converted.
    @return: Image in EXR format.
    """
    # Normalize and convert to half-float (16-bit) format
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    return image.flatten().astype(np.float16).tobytes()


def create_exr_image(individual_masks, path, base_image=None):
    """
    Creates an EXR image with painted masks as RGB layers and each mask as a separate layer.

    @param individual_masks: List of masks for each object.
    @param path: Path to save the EXR file.
    @param base_image: Base image for painting masks. If None, a blank image is used.
    """
    if base_image is None:
        # Assuming all masks have the same dimensions
        height, width = individual_masks[0].shape[:2]
        base_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Paint the masks on the image
    painted_image = base_image.copy()
    for idx, mask in enumerate(individual_masks):
        mask_color = idx + 1  # Assuming each index corresponds to a unique color
        painted_image = mask_painter(painted_image, mask, mask_color, mask_alpha=0.7, contour_color=1, contour_width=3)

    # Combine all masks for the alpha channel
    combined_alpha_mask = np.any(individual_masks, axis=0).astype(np.float32)

    # Initialize EXR file with header
    width, height = painted_image.shape[1], painted_image.shape[0]
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))

    # Define channels for RGB and additional layers for masks
    channels = {c: half_chan for c in ['R', 'G', 'B', 'A']}
    for idx in range(len(individual_masks)):
        layer_name = f'Mask{idx}'
        channels[f'{layer_name}.R'] = half_chan
        channels[f'{layer_name}.G'] = half_chan
        channels[f'{layer_name}.B'] = half_chan

    header['channels'] = channels

    # Convert RGB layers to EXR format
    rgb_layers = {c: convert_to_exr_format(painted_image[:, :, i]) for i, c in enumerate(['R', 'G', 'B'])}
    alpha_layer = convert_to_exr_format(combined_alpha_mask)

    # Convert each mask to EXR format and add as separate layers
    mask_layers = {}
    for idx, mask in enumerate(individual_masks):
        mask_exr = convert_to_exr_format(mask.astype(np.float32))
        layer_name = f'Mask{idx}'
        mask_layers[f'{layer_name}.R'] = mask_exr
        mask_layers[f'{layer_name}.G'] = mask_exr
        mask_layers[f'{layer_name}.B'] = mask_exr

    # Write layers to the EXR file
    exr_data = {**rgb_layers, 'A': alpha_layer, **mask_layers}
    path = path  if path.endswith('.exr') else path + ".exr"
    exr_file = OpenEXR.OutputFile(path, header)
    exr_file.writePixels(exr_data)
    exr_file.close()

    return


if __name__ == '__main__':
    input_image = np.array(Image.open('images/painter_input_image.jpg').convert('RGB'))
    input_mask = np.array(Image.open('images/painter_input_mask.jpg').convert('P'))

    # example of mask painter
    mask_color = 3
    mask_alpha = 0.7
    contour_color = 1
    contour_width = 5

    # save
    painted_image = Image.fromarray(input_image)
    painted_image.save('images/original.png')

    painted_image = mask_painter(input_image, input_mask, mask_color, mask_alpha, contour_color, contour_width)
    # save
    painted_image = Image.fromarray(input_image)
    painted_image.save('images/original1.png')

    # example of point painter
    input_image = np.array(Image.open('images/painter_input_image.jpg').convert('RGB'))
    input_points = np.array([[500, 375], [70, 600]])  # x, y
    point_color = 5
    point_alpha = 0.9
    point_radius = 15
    contour_color = 2
    contour_width = 5
    painted_image_1 = point_painter(input_image, input_points, point_color, point_alpha, point_radius, contour_color,
                                    contour_width)
    # save
    painted_image = Image.fromarray(painted_image_1)
    painted_image.save('images/point_painter_1.png')

    input_image = np.array(Image.open('images/painter_input_image.jpg').convert('RGB'))
    painted_image_2 = point_painter(input_image, input_points, point_color=9, point_radius=20, contour_color=29)
    # save
    painted_image = Image.fromarray(painted_image_2)
    painted_image.save('images/point_painter_2.png')

    # example of background remover
    input_image = np.array(Image.open('images/original.png').convert('RGB'))
    image_wo_background = background_remover(input_image, input_mask)  # return PIL.Image
    image_wo_background.save('images/image_wo_background.png')
