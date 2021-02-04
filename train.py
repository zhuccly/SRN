from datetime import datetime
import data_provider
from menpo.shape.pointcloud import PointCloud
import models
import numpy as np
import os.path
import slim
import tensorflow as tf
import time
import utils
import menpo
import scipy.io as sio
from tensorflow.python.framework import ops
import os        #feilong-----------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#-------------------------------------------------

#ignore  differentiable for extract_patches op
ops.NotDifferentiable("ExtractPatches")
ops.NotDifferentiable("ResizeBilinearGrad")
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0005,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('batch_size', 64, """The batch size to use.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """How many preprocess threads to use.""")
tf.app.flags.DEFINE_string('train_dir', 'ckpt/SRN',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
tf.app.flags.DEFINE_string('datasets', ':'.join(
    (
     # 'databases/lfpw/trainset/*.png',
     'databases/afw/*.jpg',
     # 'databases/helen/trainset/*.jpg',

     )),
                           """Directory where to write event logs """
                           """and checkpoint.""")
# tf.app.flags.DEFINE_string('real_data', ':'.join(
#     (
#         'databases/COFW_color/trainset/*.jpg',
#
#     )),
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
tf.app.flags.DEFINE_float('image_size', 224.

                            , 'The extracted patch size')
tf.app.flags.DEFINE_integer('patch_size', 36

                            , 'The extracted patch size')
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999




def train(scope=''):

    # H-LIU: reallocating gpu memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    """Train on dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/gpu:0'):

        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        train_dirs = FLAGS.datasets.split(':')
        # real_data = FLAGS.real_data.split(':')

        # Calculate the learning rate schedule.
        decay_steps = 1000

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)

        # Override the number of preprocessing threads to account for the increased
        # number of GPU towers.
        num_preprocess_threads = FLAGS.num_preprocess_threads

        _images, _shapes, _reference_shape, pca_model = \
            data_provider.load_images(train_dirs)
        # rela_images  = \
        #     data_provider.load_data(real_data,_reference_shape)

        reference_shape = tf.constant(_reference_shape,
                                      dtype=tf.float32,
                                      name='reference_shape')

        image_shape = _images[0].shape
        # real_shape = rela_images[0].shape
        lms_shape = _shapes[0].points.shape
        # patch = np.random.rand((3,50,50))


        def get_random_sample(rotation_stddev=10):
            image_patch = menpo.image.Image(np.random.rand(3, 256, 256))
            idx = np.random.randint(low=0, high=len(_images))
            # real_idx = np.random.randint(low=0, high=len(rela_images))
            # real_im = menpo.image.Image(rela_images[real_idx].transpose(2, 0, 1), copy=False)
            im = menpo.image.Image(_images[idx].transpose(2, 0, 1), copy=False)
            lms = _shapes[idx]
            im.landmarks['PTS'] = lms
            lms_np = np.array(lms.points)

            if np.random.rand() < .5:
               im = utils.mirror_image(im)

            if np.random.rand() < .5:
              theta = np.random.normal(scale=rotation_stddev)
              rot = menpo.transform.rotate_ccw_about_centre(lms, theta)
              im = im.warp_to_shape(im.shape, rot)

            index = np.random.randint(0,67)
            center = lms_np[index:index+1,:].reshape(1,2)
            # print center
            center_min = (50-center.copy())>0
            center_max = (center.copy()+50)>FLAGS.image_size
            center = center-center_max*50
            # print center+center_min*50
            center = PointCloud(center+center_min*50)
            # center=PointCloud(center)

            images_patchs = image_patch.extract_patches(center,(48,48))
            imm = utils.set_patches(im,images_patchs,center)

            pixels = im.pixels.transpose(1, 2, 0).astype('float32')
            pixels_occ = imm.pixels.transpose(1, 2, 0).astype('float32')
            # real_pixels = real_im.pixels.transpose(1, 2, 0).astype('float32')
            shape = im.landmarks['PTS'].lms.points.astype('float32')
            return pixels_occ, shape

        image, shape = tf.py_func(get_random_sample, [],
                                  [tf.float32, tf.float32], stateful=True)

        initial_shape = data_provider.random_shape(shape, reference_shape,
                                                   pca_model)
        image.set_shape(image_shape)
        # ims_oc.set_shape(ims_oc)
        shape.set_shape(lms_shape)
        initial_shape.set_shape(lms_shape)
        # initial_shape = initial_shape*(198./224.)

        do_scale = tf.random_uniform([1])
        # image_height = tf.to_int32(tf.to_float(FLAGS.image_size) * do_scale[0])
        # image_width = tf.to_int32(tf.to_float(FLAGS.image_size) * do_scale[0])
        # image = tf.image.resize_images(image, tf.stack([image_height, image_width]))
        # shape = shape * do_scale
        # initial_shape = initial_shape * do_scale


        target_h = tf.to_int32(198)
        target_w = tf.to_int32(198)
        offset_h = tf.to_int32((tf.to_int32(FLAGS.image_size) - target_h) / 2)
        offset_w = tf.to_int32((tf.to_int32(FLAGS.image_size) - target_w) / 2)
        offset_h = tf.to_int32(tf.to_float(offset_h) * do_scale[0])
        offset_w = tf.to_int32(tf.to_float(offset_w) * do_scale[0])
        image = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, target_h, target_w)
        shape = shape - tf.to_float(tf.stack([offset_h, offset_w]))
        initial_shape = initial_shape - tf.to_float(tf.stack([offset_h, offset_w]))



        # with tf.device(FLAGS.train_device):

        image = data_provider.distort_color(image)

        images, lms, inits = tf.train.batch([image, shape, initial_shape],
                                            FLAGS.batch_size,
                                            dynamic_pad=False,
                                            capacity=5000,
                                            enqueue_many=False,
                                            num_threads=num_preprocess_threads,
                                            name='batch')
        print('Defining model...')
        with tf.device(FLAGS.train_device):

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            predictions, dxs, _ = models.model(images, inits, patch_shape=(FLAGS.patch_size, FLAGS.patch_size))

            # _, dxs_g, _ = mdm_model.model(g_images, inits, patch_shape=(FLAGS.patch_size, FLAGS.patch_size),reuse=tf.AUTO_REUSE)

            total_loss = 0

            lmss = []
            lmss.append(lms)

            for i, dx in enumerate(dxs):
                norm_error = models.normalized_rmse(dx + inits, lms)
                tf.summary.histogram('errors', norm_error)
                loss = tf.reduce_mean(norm_error)
                total_loss += loss
                summaries.append(tf.summary.scalar('losses/step_{}'.format(i),
                                                   loss))


            grads = opt.compute_gradients(total_loss)

        summaries.append(tf.summary.scalar('losses/total', total_loss))



        summary = tf.summary.image('images',
                                   tf.concat([images],2),
                                   5)
        summaries.append(tf.summary.histogram('dx', predictions - inits))

        summaries.append(summary)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                              scope)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name +
                                                      '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

        # Another possibility is to use tf.slim.get_variables().
        variables_to_average = (
            tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        # NOTE: Currently we are not using batchnorm in MDM.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())


        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)
        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
       # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        print('Initializing variables...')
        sess.run(init)
        print('Initialized variables.')


        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir)

        print('Starting training...')
        for step in range(FLAGS.max_steps):

            start_time = time.time()
            _, loss_value = sess.run([train_op, total_loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
