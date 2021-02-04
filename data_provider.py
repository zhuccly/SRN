from functools import partial
from menpo.shape.pointcloud import PointCloud
from menpofit.builder import compute_reference_shape
from menpofit.builder import rescale_images_to_reference_shape
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
from pathlib import Path

import joblib
import menpo.feature
import menpo.image
import menpo.io as mio
import numpy as np
import tensorflow as tf
import detect
import utils
from menpo.transform import Translation, Scale
from menpo.shape import PointCloud
FLAGS = tf.app.flags.FLAGS

def build_reference_shape(paths, diagonal=200):
    """Builds the reference shape.
    Args:
      paths: paths that contain the ground truth landmark files.
      diagonal: the diagonal of the reference shape in pixels.
    Returns:
      the reference shape.
    """
    landmarks = []
    for path in paths:
        path = Path(path).parent.as_posix()

        landmarks += [
            group.lms
            for group in mio.import_landmark_files(path, verbose=True)
            if group.lms.n_points == 68
        ]

    return compute_reference_shape(landmarks,
                                   diagonal=diagonal).points.astype(np.float32)


def grey_to_rgb(im):
    """Converts menpo Image to rgb if greyscale
    Args:
      im: menpo Image with 1 or 3 channels.
    Returns:
      Converted menpo `Image'.
    """
    assert im.n_channels in [1, 3]

    if im.n_channels == 3:
        return im

    im.pixels = np.vstack([im.pixels] * 3)
    return im


def align_reference_shape(reference_shape, bb):
    min_xy = tf.reduce_min(reference_shape, 0)
    max_xy = tf.reduce_max(reference_shape, 0)
    min_x, min_y = min_xy[0], min_xy[1]
    max_x, max_y = max_xy[0], max_xy[1]

    reference_shape_bb = tf.stack([[min_x, min_y], [max_x, min_y],
                                  [max_x, max_y], [min_x, max_y]])

    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    return tf.add(
        (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio,
        tf.reduce_mean(bb, 0),
        name='initial_shape')


def random_shape(gts, reference_shape, pca_model):
    """Generates a new shape estimate given the ground truth shape.
    Args:
      gts: a numpy array [num_landmarks, 2]
      reference_shape: a Tensor of dimensions [num_landmarks, 2]
      pca_model: A PCAModel that generates shapes.
    Returns:
      The aligned shape, as a Tensor [num_landmarks, 2].
    """

    def synthesize(lms):
        return detect.synthesize_detection(pca_model, menpo.shape.PointCloud(
            lms).bounding_box()).points.astype(np.float32)

    bb, = tf.py_func(synthesize, [gts], [tf.float32])
    shape = align_reference_shape(reference_shape, bb)
    shape.set_shape(reference_shape.get_shape())

    return shape


def get_noisy_init_from_bb(reference_shape, bb, noise_percentage=.02):
    """Roughly aligns a reference shape to a bounding box.
    This adds some uniform noise for translation and scale to the
    aligned shape.
    Args:
      reference_shape: a numpy array [num_landmarks, 2]
      bb: bounding box, a numpy array [4, ]
      noise_percentage: noise presentation to add.
    Returns:
      The aligned shape, as a numpy array [num_landmarks, 2]
    """
    bb = PointCloud(bb)
    reference_shape = PointCloud(reference_shape)

    bb = noisy_shape_from_bounding_box(
        reference_shape,
        bb,
        noise_percentage=[noise_percentage, 0, noise_percentage]).bounding_box(
        )

    return align_shape_with_bounding_box(reference_shape, bb).points


def load_images(paths, group=None, verbose=True):
    """Loads and rescales input images to the diagonal of the reference shape.
    Args:
      paths: a list of strings containing the data directories.
      reference_shape: a numpy array [num_landmarks, 2]
      group: landmark group containing the grounth truth landmarks.
      verbose: boolean, print debugging info.
    Returns:
      images: a list of numpy arrays containing images.
      shapes: a list of the ground truth landmarks.
      reference_shape: a numpy array [num_landmarks, 2].
      shape_gen: PCAModel, a shape generator.
    """
    images = []
    shapes = []
    bbs = []

    reference_shape = PointCloud(build_reference_shape(paths))
    # print(reference_shape.lms.points.shape)
    # train_dir = Path(FLAGS.train_dir)
    # reference_shape = PointCloud(mio.import_pickle(train_dir / 'reference_shape.pkl'))
    # print(reference_shape.shape)


    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))

        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            group = 'PTS'#group or im.landmarks[group]._group_label

            bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
            if 'set' not in str(bb_root):
                bb_root = im.path.parent.relative_to(im.path.parent.parent)
            #print(str(Path('bbs2')))
            #load bounding box
            im.landmarks['bb'] = mio.import_landmark_file(str(Path(
                'bbs') / bb_root / (im.path.stem + '.pts')))
            # crop

            # im = im.crop_to_landmarks_proportion(0.3, group='bb')
            im,trans = crop_image_bounding_box(im, im.landmarks['bb'], [FLAGS.image_size, FLAGS.image_size], base=198./FLAGS.image_size, order=1)
            ini_shape = PointCloud(trans.apply(reference_shape.points.copy()))
            # im.view()
            # im.view_landmarks(group='PTS')
            # print reference_shape.points
            # im = im.rescale_to_pointcloud(reference_shape, group=group)
            # im = menpo.image.Image(crop_i.pixels_with_channels_at_back())
            im = grey_to_rgb(im)
            images.append(im.pixels.transpose(1, 2, 0))
            shapes.append(im.landmarks[group].lms)
            bbs.append(im.landmarks['bb'].lms)

    train_dir = Path(FLAGS.train_dir)
    mio.export_pickle(reference_shape.points, train_dir / 'reference_shape.pkl', overwrite=True)
    print('created reference_shape.pkl using the {} group'.format(group))

    pca_model = detect.create_generator(shapes, bbs)

    # Pad images to max length
    max_shape = np.max([im.shape for im in images], axis=0)
    max_shape = [len(images)] + list(max_shape)
    padded_images = np.random.rand(*max_shape).astype(np.float32)
    print(padded_images.shape)

    for i, im in enumerate(images):
        print ("======================================================================")
        print (im.shape)
        height, width = im.shape[:2]
        dy = max(int((max_shape[1] - height - 1) / 2), 0)
        dx = max(int((max_shape[2] - width - 1) / 2), 0)
        lms = shapes[i]
        pts = lms.points
        pts[:, 0] += dy
        pts[:, 1] += dx

        lms = lms.from_vector(pts)
        padded_images[i, dy:(height+dy), dx:(width+dx)] = im

    return padded_images, shapes, ini_shape.points, pca_model


def load_data(paths, reference_shape, verbose=True):
    """Loads and rescales input images to the diagonal of the reference shape.
    Args:
      paths: a list of strings containing the data directories.
      reference_shape: a numpy array [num_landmarks, 2]
      group: landmark group containing the grounth truth landmarks.
      verbose: boolean, print debugging info.
    Returns:
      images: a list of numpy arrays containing images.
      shapes: a list of the ground truth landmarks.
      reference_shape: a numpy array [num_landmarks, 2].
      shape_gen: PCAModel, a shape generator.
    """
    images = []
    shapes = []
    bbs = []

    # print(reference_shape.lms.points.shape)
    # train_dir = Path(FLAGS.train_dir)
    # reference_shape = PointCloud(mio.import_pickle(train_dir / 'reference_shape.pkl'))
    # print(reference_shape.shape)


    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))

        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            group = 'PTS'#group or im.landmarks[group]._group_label

            bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
            if 'set' not in str(bb_root):
                bb_root = im.path.parent.relative_to(im.path.parent.parent)
            #print(str(Path('bbs2')))
            #load bounding box
            im.landmarks['bb'] = mio.import_landmark_file(str(Path(
                'bbs') / bb_root / (im.path.stem + '.pts')))
            # crop
            im,trans = crop_image_bounding_box(im, im.landmarks['bb'], [224., 224.], base=198./224., order=1)
            # im = im.crop_to_landmarks_proportion(0.3, group='bb')
            # im = im.rescale_to_pointcloud(reference_shape, group=group)
            im = grey_to_rgb(im)
            images.append(im.pixels.transpose(1, 2, 0))
            # shapes.append(im.landmarks[group].lms)


    # train_dir = Path(FLAGS.train_dir)
    # mio.export_pickle(reference_shape.points, train_dir / 'reference_shape.pkl', overwrite=True)
    # print('created reference_shape.pkl using the {} group'.format(group))
    #
    # pca_model = detect.create_generator(shapes, bbs)

    # Pad images to max length
    max_shape = np.max([im.shape for im in images], axis=0)
    max_shape = [len(images)] + list(max_shape)
    padded_images = np.random.rand(*max_shape).astype(np.float32)
    print(padded_images.shape)

    for i, im in enumerate(images):
        print ("======================================================================")
        print (im.shape)
        height, width = im.shape[:2]
        dy = max(int((max_shape[1] - height - 1) / 2), 0)
        dx = max(int((max_shape[2] - width - 1) / 2), 0)


        padded_images[i, dy:(height+dy), dx:(width+dx)] = im

    return padded_images
def load_image(path, reference_shape, is_training=False, group='PTS',
               mirror_image=False):
    """Load an annotated image.
    In the directory of the provided image file, there
    should exist a landmark file (.pts) with the same
    basename as the image file.
    Args:
      path: a path containing an image file.
      reference_shape: a numpy array [num_landmarks, 2]
      is_training: whether in training mode or not.
      group: landmark group containing the grounth truth landmarks.
      mirror_image: flips horizontally the image's pixels and landmarks.
    Returns:
      pixels: a numpy array [width, height, 3].
      estimate: an initial estimate a numpy array [68, 2].
      gt_truth: the ground truth landmarks, a numpy array [68, 2].
    """
    im = mio.import_image(path)

    bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
    if 'set' not in str(bb_root):
        bb_root = im.path.parent.relative_to(im.path.parent.parent)
    # print (str(Path('bbs2') / bb_root / (
    #     im.path.stem + '.pts')))

    # im.landmarks['bb'] = mio.import_landmark_file(str(Path('bbs') / bb_root / (im.path.stem.replace(' ', '') + '.pts')))
    #
    # im = im.crop_to_landmarks_proportion(0.3, group='bb')
    # reference_shape = PointCloud(reference_shape)
    #
    # bb = im.landmarks['bb'].lms.bounding_box()
    #
    # im.landmarks['__initial'] = align_shape_with_bounding_box(reference_shape,
    #                                                           bb)
    # im = im.rescale_to_pointcloud(reference_shape, group='__initial')
    im.landmarks['bb'] = mio.import_landmark_file(str(Path('bbs') / bb_root / (im.path.stem.replace(' ', '') + '.pts')))

    # im = im.crop_to_landmarks_proportion(0.3, group='bb')
    im, trans = crop_image_bounding_box(im, im.landmarks['bb'], [224., 224.], base=1, order=1)
    # ini_shape = PointCloud(trans.apply(reference_shape.points.copy()))
    ini_shape = PointCloud(trans.apply(reference_shape.copy()))

    bb = im.landmarks['bb'].lms.bounding_box()

    im.landmarks['__initial'] = align_shape_with_bounding_box(ini_shape,
                                                              bb)
    if mirror_image:
        im = utils.mirror_image(im)

    lms = im.landmarks[group].lms
    initial = im.landmarks['__initial'].lms

    # if the image is greyscale then convert to rgb.
    pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)

    gt_truth = lms.points.astype(np.float32)
    estimate = initial.points.astype(np.float32)
    return pixels.astype(np.float32).copy(), gt_truth, estimate


def distort_color(image, thread_id=0, stddev=0.1, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        image += tf.random_normal(
                tf.shape(image),
                stddev=stddev,
                dtype=tf.float32,
                seed=42,
                name='add_gaussian_noise')
        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def batch_inputs(paths,
                 reference_shape,
                 batch_size=32,
                 is_training=False,
                 num_landmarks=68,
                 mirror_image=False):
    """Reads the files off the disk and produces batches.
    Args:
      paths: a list of directories that contain training images and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_traininig: whether in training mode.
      num_landmarks: the number of landmarks in the training images.
      mirror_image: mirrors the image and landmarks horizontally.
    Returns:
      images: a tf tensor of shape [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, 68, 2].
      lms_init: a tf tensor of shape [batch_size, 68, 2].
    """

    files = tf.concat([map(str, sorted(Path(d).parent.glob(Path(d).name)))
                          for d in paths], 0)
    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=is_training,
                                                    capacity=5000)

    filename = filename_queue.dequeue()
    print (filename)
    image, lms, lms_init = tf.py_func(
        partial(load_image, is_training=is_training,
                mirror_image=mirror_image),
        [filename, reference_shape], # input arguments
        [tf.float32, tf.float32, tf.float32], # output types
        name='load_image'
    )

    # The image has always 3 channels.
    image.set_shape([None, None, 3])

    if is_training:
        image = distort_color(image)

    lms = tf.reshape(lms, [num_landmarks, 2])
    lms_init = tf.reshape(lms_init, [num_landmarks, 2])

    images, lms, inits, shapes = tf.train.batch(
                                    [image, lms, lms_init, tf.shape(image)],
                                    batch_size=batch_size,
                                    num_threads=1 if is_training else 1,
                                    capacity=5000,
                                    enqueue_many=False,
				    dynamic_pad=True)

    return images, lms, inits, shapes

def crop_image_bounding_box(img, bbox, res, base=256., order=1):

    center = bbox.centre()
    bmin, bmax = bbox.bounds()
    scale = np.linalg.norm(bmax - bmin) / base

    return crop_image(img, center, scale, res, base, order=order)
def crop_image(img, center, scale, res, base=256., order=1):
    h = scale

    t = Translation(
        [
            res[0] * (-center[0] / h + .5),
            res[1] * (-center[1] / h + .5)
        ]).compose_after(Scale((res[0] / h, res[1] / h))).pseudoinverse()

    # Upper left point
    ul = np.floor(t.apply([0, 0]))
    # Bottom right point
    br = np.ceil(t.apply(res).astype(np.int))

    # crop and rescale
    cimg, trans = img.warp_to_shape(
        br - ul, Translation(-(br - ul) / 2 + (br + ul) / 2), return_transform=True)

    c_scale = np.min(cimg.shape) / np.mean(res)
    new_img = cimg.rescale(1 / c_scale, order=order).resize(res, order=order)

    trans = trans.compose_after(Scale([c_scale, c_scale]))

    return new_img, trans
