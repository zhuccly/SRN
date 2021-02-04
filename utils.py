import numpy as np
from menpo.shape import PointCloud
import cv2

jaw_indices = np.arange(0, 17)
lbrow_indices = np.arange(17, 22)
rbrow_indices = np.arange(22, 27)
upper_nose_indices = np.arange(27, 31)
lower_nose_indices = np.arange(31, 36)
leye_indices = np.arange(36, 42)
reye_indices = np.arange(42, 48)
outer_mouth_indices = np.arange(48, 60)
inner_mouth_indices = np.arange(60, 68)

parts_68 = (jaw_indices, lbrow_indices, rbrow_indices, upper_nose_indices,
            lower_nose_indices, leye_indices, reye_indices,
            outer_mouth_indices, inner_mouth_indices)

mirrored_parts_68 = np.hstack([
    jaw_indices[::-1], rbrow_indices[::-1], lbrow_indices[::-1],
    upper_nose_indices, lower_nose_indices[::-1],
    np.roll(reye_indices[::-1], 4), np.roll(leye_indices[::-1], 4),
    np.roll(outer_mouth_indices[::-1], 7),
    np.roll(inner_mouth_indices[::-1], 5)
])


def mirror_landmarks_68(lms, image_size):
    return PointCloud(abs(np.array([0, image_size[1]]) - lms.as_vector(
    ).reshape(-1, 2))[mirrored_parts_68])


def mirror_image(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1].copy()

    for group in im.landmarks:
        lms = im.landmarks[group].lms
        if lms.points.shape[0] == 68:
            im.landmarks[group] = mirror_landmarks_68(lms, im.shape)

    return im


def mirror_image_bb(im):
    im = im.copy()
    im.pixels = im.pixels[..., ::-1]
    im.landmarks['bounding_box'] = PointCloud(abs(np.array([0, im.shape[
        1]]) - im.landmarks['bounding_box'].lms.points))
    return im


def line(image, x0, y0, x1, y1, color):
    steep = False
    if x0 < 0 or x0 >= 400 or x1 < 0 or x1 >= 400 or y0 < 0 or y0 >= 400 or y1 < 0 or y1 >= 400:
        return

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1) + 1):
        t = (x - x0) / float(x1 - x0)
        y = y0 * (1 - t) + y1 * t
        if steep:
            image[x, int(y)] = color
        else:
            image[int(y), x] = color
            
def point(image, x0, y0, color):
    if x0 < 0 or x0 >= 400 or y0 < 0 or y0 >= 400:
        return
    cv2.circle(image,(y0,x0),3,(0.5,0.5,0.5),-1)
    cv2.circle(image,(y0,x0),2,(1,1,1),-1)
#    for x in range(int(x0)-2, int(x0) + 3):
#        for y in range(int(y0)-2, int(y0) + 3):
#            if((x-x0)**2+(y-y0)**2<=4):
#                image[x, y] = color


def draw_landmarks(img, lms):
    try:
        img = img.copy()

        for i, part in enumerate(parts_68[1:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]

                line(img, p2[1], p2[0], p1[1], p1[0], 1)
    except:
        pass
    return img
    
def draw_landmarks_point(img, lms):
    try:
        img = img.copy()

        for i in range(lms.shape[0]):
            point(img,lms[i][0],lms[i][1],1)
    except:
        pass
    return img


def batch_draw_landmarks(imgs, lms):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, lms)])
    
def batch_draw_landmarks_point(imgs, lms):
    return np.array([draw_landmarks_point(img, l) for img, l in zip(imgs, lms)])


def get_central_crop(images, box=(6, 6)):
    _, w, h, _ = images.get_shape().as_list()

    half_box = (box[0] / 2., box[1] / 2.)

    a = slice(int((w // 2) - half_box[0]), int((w // 2) + half_box[0]))
    b = slice(int((h // 2) - half_box[1]), int((h // 2) + half_box[1]))

    return images[:, a, b, :]


def build_sampling_grid(patch_shape):
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)


default_sampling_grid = build_sampling_grid((30, 30))


def extract_patches(pixels, centres, sampling_grid=default_sampling_grid):
    """ Extracts patches from an image.

    Args:
        pixels: a numpy array of dimensions [width, height, channels]
        centres: a numpy array of dimensions [num_patches, 2]
        sampling_grid: (patch_width, patch_height, 2)

    Returns:
        a numpy array [num_patches, width, height, channels]
    """
    pixels = pixels.transpose(2, 0, 1)

    max_x = pixels.shape[-2] - 1
    max_y = pixels.shape[-1] - 1

    patch_grid = (sampling_grid[None, :, :, :] + centres[:, None, None, :]
                  ).astype('int32')

    X = patch_grid[:, :, :, 0].clip(0, max_x)
    Y = patch_grid[:, :, :, 1].clip(0, max_y)

    return pixels[:, X, Y].transpose(1, 2, 3, 0)

def set_patches(image, patches, patch_centers, offset=None, offset_index=None):
    r"""
    Parameters
    ----------
    patches : `ndarray` or `list`
        The values of the patches.
        A ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
    patch_centers : :map:`PointCloud`
        The centers to set the patches around.
    offset : `list` or `tuple` or ``(1, 2)`` `ndarray`
        The offset to apply on the patch centers within the image.
    offset_index : `int`
        The offset index within the provided `patches` argument, thus the
        index of the second dimension from which to sample.
    Raises
    ------
    ValueError
        If pixels array is not 2D
    """
    # if self.ndim != 3:
    #     raise ValueError(
    #         "Only 2D images are supported but " "found {}".format(self.shape)
    #     )
    if offset is None:
        offset = np.zeros([1, 2], dtype=np.intp)
    # elif isinstance(offset, tuple) or isinstance(offset, list):
    #     offset = np.asarray([offset])
    # offset = np.require(offset, dtype=np.intp)
    if offset_index is None:
        offset_index = 0

    copy = image.copy()
    # set patches
    set_patch(patches, copy.pixels, patch_centers.points, offset, offset_index)
    return copy

def set_patch(patches, pixels, patch_centers, offset, offset_index):
    r"""
    Set the values of a group of patches into the correct regions of a copy
    of this image. Given an array of patches and a set of patch centers,
    the patches' values are copied in the regions of the image that are
    centred on the coordinates of the given centers.
    The patches argument can have any of the two formats that are returned
    from the `extract_patches()` and `extract_patches_around_landmarks()`
    methods. Specifically it can be:
        1. ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
        2. `list` of ``n_center * n_offset`` :map:`Image` objects
    Currently only 2D images are supported.
    Parameters
    ----------
    patches : `ndarray` or `list`
        The values of the patches.
        A ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
    pixels : ``(n_channels, height, width)`` `ndarray``
        Pixel array to replace the patches within
    patch_centers : :map:`PointCloud`
        The centers to set the patches around.
    offset : `list` or `tuple` or ``(1, 2)`` `ndarray`
        The offset to apply on the patch centers within the image.
    offset_index : `int`
        The offset index within the provided `patches` argument, thus the
        index of the second dimension from which to sample.
    Raises
    ------
    ValueError
        If pixels array is not 2D
    """
    if pixels.ndim != 3:
        raise ValueError(
            "Only 2D images are supported but " "found {}".format(pixels.shape)
        )

    patch_shape = patches.shape[-2:]
    # the [L]ow offset is the floor of half the patch shape
    l_r, l_c = (int(patch_shape[0] // 2), int(patch_shape[1] // 2))
    # the [H]igh offset needs to be one pixel larger if the original
    # patch was odd
    h_r, h_c = (int(l_r + patch_shape[0] % 2), int(l_c + patch_shape[1] % 2))
    for patches_with_offsets, point in zip(patches, patch_centers):
        patch = patches_with_offsets[offset_index]
        p = point + offset[0]
        p_r = int(p[0])
        p_c = int(p[1])
        pixels[:, p_r - l_r : p_r + h_r, p_c - l_c : p_c + h_c] = patch