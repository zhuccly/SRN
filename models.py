from functools import partial

import slim
import tensorflow as tf


from slim import ops
from slim import scopes



def align_reference_shape(reference_shape, reference_shape_bb, im, bb):
    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    align_mean_shape = (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio + tf.reduce_mean(bb, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)
    return tf.image.resize_bilinear(tf.expand_dims(im, 0), new_size)[0, :, :, :], align_mean_shape / ratio, ratio

def normalized_rmse(pred, gt_truth):
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :])**2), 1))

    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 68)


def conv_model(inputs, is_training=True, scope=''):

  # summaries or losses.
  net = {}

  with tf.name_scope(scope, 'Conv_lay', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
      with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
        net['conv_1'] = ops.conv2d(inputs, 32, [7, 7], scope='conv_1')
        net['pool_1'] = ops.max_pool(net['conv_1'], [2, 2])
        net['conv_2'] = ops.conv2d(net['pool_1'], 64, [3, 3], scope='conv_2')
        net['pool_2'] = ops.max_pool(net['conv_2'], [2, 2])
        net['conv_3'] = ops.conv2d(net['pool_2'], 64, [3, 3], scope='conv_3')
        net['pool_3'] = ops.max_pool(net['conv_3'], [2, 2])
        net['concat'] = net['pool_3']
  return net


def model(images, inits, num_iterations=3, num_patches=68, patch_shape=(36, 36), num_channels=3,reuse = False):
    batch_size = images.get_shape().as_list()[0]
    # print(batch_size)

    hiden = tf.zeros((batch_size, 512))

    dx = tf.zeros((batch_size, num_patches, 2))
    endpoints = {}
    dxs = []
    m_module = tf.load_op_library('./extract_patches.so')
    with tf.variable_scope('models', reuse=reuse):
        for step in range(num_iterations):
          with tf.device('/cpu:0'):
            patches = m_module.extract_patches(images, tf.constant(patch_shape), inits+dx)
          patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))


          endpoints['patches'] = patches

          with tf.variable_scope('convnet', reuse=step>0):
              net = conv_model(patches)
              ims = net['concat']

              num,h,w,c = net['concat'].get_shape().as_list()
              ims_all = tf.reshape(ims, (batch_size, -1))
              ims_1 = slim.ops.conv2d(net['concat'] , 64, [1,1], scope='cita')
              ims_2 = slim.ops.conv2d(net['concat'] , 64, [1, 1], scope='feita')
              ims_3 = slim.ops.conv2d(net['concat'], 64, [1, 1], scope='gama')
              ims_1 = tf.reshape(ims_1, (batch_size, -1, 64))
              ims_2 = tf.reshape(ims_2, (batch_size, 64, -1))
              ims_3 = tf.reshape(ims_3, (batch_size, -1, 64))
              ims_4 = tf.matmul(ims_1, ims_2)
              ims_4 = tf.nn.softmax(ims_4)
              ims_5 = tf.matmul(ims_4, ims_3)
              ims_5 = tf.reshape(ims_5, (batch_size*num_patches, h, w, 64))/(num*h*w*c)
              ims_5 = slim.ops.conv2d(ims_5, c, [1, 1], scope='beata')
              ims_6 = ims_5+net['concat']
          # ims_6 = tf.reshape(ims_6, (batch_size, -1, 16))
              ims_fllaten = tf.reshape(ims_6, (batch_size, num_patches, -1))
              ims_d = ims_fllaten[:,0:17,:]
              ims_d = tf.reshape(ims_d, (batch_size, -1))
              ims_t1 = ims_fllaten[:, 17:27, :]
              ims_t1 = tf.reshape(ims_t1, (batch_size, -1))
              ims_t2 = ims_fllaten[:, 36:48, :]
              ims_t2 = tf.reshape(ims_t2, (batch_size, -1))
              ims_m1 = ims_fllaten[:, 27:36, :]
              ims_m1 = tf.reshape(ims_m1, (batch_size, -1))
              ims_m2 = ims_fllaten[:, 48:68, :]
              ims_m2 = tf.reshape(ims_m2, (batch_size, -1))


          with tf.variable_scope('model', reuse=step>0) as scope:
              hiden = slim.ops.fc(tf.concat([ims_all, hiden], 1), 512, scope='rnn', activation=tf.tanh)
              top_f = slim.ops.fc(tf.concat([ims_t1,ims_t2], 1), 512, scope='top', activation=tf.tanh)
              mid_f = slim.ops.fc(tf.concat([ims_m1,ims_m2], 1), 512, scope='mid', activation=tf.tanh)
              down_f = slim.ops.fc(tf.concat([ims_d], 1), 512, scope='down', activation=tf.tanh)

              brows_f = slim.ops.fc(tf.concat([top_f, mid_f,hiden], 1), 256, scope='brow')
              eyes = slim.ops.fc(tf.concat([top_f, mid_f,hiden], 1), 256, scope='eye')
              nose = slim.ops.fc(tf.concat([top_f, mid_f, down_f,hiden], 1), 256, scope='nose')
              mouth = slim.ops.fc(tf.concat([top_f, mid_f, down_f,hiden], 1), 256, scope='mouth')
              l = slim.ops.fc(tf.concat([mid_f, down_f,hiden], 1), 256, scope='l')
              r = slim.ops.fc(tf.concat([mid_f, down_f,hiden], 1), 256, scope='r')

              b_p = slim.ops.fc(tf.concat([brows_f, eyes], 1), 10, scope='brow_p', activation=None)
              e_p = slim.ops.fc(tf.concat([brows_f, eyes], 1), 12, scope='e_p', activation=None)
              n_p = slim.ops.fc(tf.concat([nose, mouth], 1), 9, scope='n_p', activation=None)
              m_p = slim.ops.fc(tf.concat([nose, mouth], 1), 20, scope='m_p', activation=None)
              l_p = slim.ops.fc(tf.concat([l, r], 1), 9, scope='l_p', activation=None)
              r_p = slim.ops.fc(tf.concat([l, r], 1), 8, scope='r_p', activation=None)
              rela = tf.concat([l_p, r_p, b_p, n_p, e_p, m_p], 1)
              prediction_full = slim.ops.fc(rela, 68 * 2, scope='full', activation=None)

              endpoints['prediction'] = prediction_full
          prediction = tf.reshape(prediction_full, (batch_size, num_patches, 2))
          dx += prediction
          dxs.append(dx)



    return inits + dx, dxs, endpoints