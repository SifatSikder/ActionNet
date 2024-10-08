from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..correlation import correlation

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_resnet_v2_base(inputs,
                             final_endpoint='Conv2d_7b_1x1',
                             output_stride=16,
                             align_feature_maps=False,
                             scope=None,
                             activation_fn=tf.nn.relu):

  if output_stride != 8 and output_stride != 16:
    raise ValueError('output_stride must be 8 or 16.')

  padding = 'SAME' if align_feature_maps else 'VALID'

  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 149 x 149 x 32
      net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,
                        scope='Conv2d_1a_3x3')
      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points

      # 147 x 147 x 32
      net = slim.conv2d(net, 32, 3, padding=padding,
                        scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      # 147 x 147 x 64
      net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      # 73 x 73 x 64
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_3a_3x3')
      if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
      # 73 x 73 x 80
      net = slim.conv2d(net, 80, 1, padding=padding,
                        scope='Conv2d_3b_1x1')
      if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
      # 71 x 71 x 192
      net = slim.conv2d(net, 192, 3, padding=padding,
                        scope='Conv2d_4a_3x3')
      if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
      # 35 x 35 x 192
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_5a_3x3')
      if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points

      # 35 x 35 x 320
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                      scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                      scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                       scope='AvgPool_0a_3x3')
          tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                     scope='Conv2d_0b_1x1')
        net = tf.concat(
            [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

      if add_and_check_final('Mixed_5b', net): return net, end_points
      # TODO(alemi): Register intermediate endpoints
      net = slim.repeat(net, 10, block35, scale=0.17,
                        activation_fn=activation_fn)

      # 17 x 17 x 1088 if output_stride == 8,
      # 33 x 33 x 1088 if output_stride == 16
      use_atrous = output_stride == 8

      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                   padding=padding,
                                   scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                      stride=1 if use_atrous else 2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

      if add_and_check_final('Mixed_6a', net): return net, end_points

      # TODO(alemi): register intermediate endpoints
      with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
        net = slim.repeat(net, 20, block17, scale=0.10,
                          activation_fn=activation_fn)
      if add_and_check_final('PreAuxLogits', net): return net, end_points

      if output_stride == 8:
        # TODO(gpapan): Properly support output_stride for the rest of the net.
        raise ValueError('output_stride==8 is only supported up to the '
                         'PreAuxlogits end_point for now.')

      # 8 x 8 x 2080
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                     padding=padding,
                                     scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.max_pool2d(net, 3, stride=2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat(
            [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

      if add_and_check_final('Mixed_7a', net): return net, end_points

      # TODO(alemi): register intermediate endpoints
      net = slim.repeat(net, 9, block8, scale=0.20, activation_fn=activation_fn)
      net = block8(net, activation_fn=None)

      # 8 x 8 x 1536
      net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
      if add_and_check_final('Conv2d_7b_1x1', net): return net, end_points

    raise ValueError('final_endpoint (%s) not recognized', final_endpoint)


def flownet_s(input_a, input_b, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2',
                        create_aux_logits=True,
                        activation_fn=tf.nn.relu):

  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [input_a, input_b],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      concat_inputs = tf.concat([input_a, input_b], axis=3)

      net, end_points = inception_resnet_v2_base(concat_inputs, scope=scope,
                                                 activation_fn=activation_fn)

      if create_aux_logits and num_classes:
        with tf.variable_scope('AuxLogits'):
          aux = end_points['PreAuxLogits']
          aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
                                scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux

      with tf.variable_scope('Logits'):
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        kernel_size = net.get_shape()[1:3]
        if kernel_size.is_fully_defined():
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_8x8')
        else:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
        if not num_classes:
          return net, end_points
        net = slim.flatten(net)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')
        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points
flownet_s.default_image_size = 299

def flownet_si(input_a, input_b, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=tf.AUTO_REUSE,
                        scope='InceptionResnetV2',
                        create_aux_logits=True,
                        activation_fn=tf.nn.relu):

  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [input_a, input_b],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net_a, end_points_a = inception_resnet_v2_base(input_a, scope=scope,
                                                 activation_fn=activation_fn)
      net_b, end_points_b = inception_resnet_v2_base(input_b, scope=scope,
                                                 activation_fn=activation_fn)
      print (net_a)
      print (net_b)
      net = tf.concat([net_a, net_b], axis=3)
      for keys in end_points_a.keys():
        end_points[keys] = tf.concat([end_points_a[keys], end_points_b[keys]], axis=3)

      if create_aux_logits and num_classes:
        with tf.variable_scope('AuxLogits'):
          aux = end_points['PreAuxLogits']
          aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
                                scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux

      with tf.variable_scope('Logits'):
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        kernel_size = net.get_shape()[1:3]
        if kernel_size.is_fully_defined():
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_8x8')
        else:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
        if not num_classes:
          return net, end_points
        net = slim.flatten(net)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')
        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points
flownet_si.default_image_size = 299

def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001,
                                  activation_fn=tf.nn.relu):

  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'fused': None,  # Use fused batch norm if possible.
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope


def flownet_arg_scope(weight_decay=0.0005):

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def flownet_s_vgg(input_a,
           input_b,
           num_classes=9,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='flownet_s',
           fc_conv_padding='VALID',
           global_pool=False):

  with tf.variable_scope(scope, 'flownet_s', [input_a, input_b]) as sc:
    concat_inputs = tf.concat([input_a, input_b], axis=3)

    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(concat_inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
#flownet_s.default_image_size = 224

def flownet_c(input_a, input_b, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2',
                        create_aux_logits=True,
                        activation_fn=tf.nn.relu):

  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [input_a, input_b],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):

      conv_a = slim.repeat(input_a, 3, slim.conv2d, 64, [3,3], scope = 'conv_a')
      conv_b = slim.repeat(input_b, 3, slim.conv2d, 64, [3,3], scope = 'conv_b')
      cc = correlation(conv_a, conv_b, 1, 20, 1, 2, 20)

      conv_a_4 = slim.conv2d(conv_a, 32, 1, scope='conv_redir')
      concat_inputs = tf.concat([conv_a_4, cc], axis=3)

      net, end_points = inception_resnet_v2_base(concat_inputs, scope=scope,
                                                 activation_fn=activation_fn)

      if create_aux_logits and num_classes:
        with tf.variable_scope('AuxLogits'):
          aux = end_points['PreAuxLogits']
          aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
                                scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux

      with tf.variable_scope('Logits'):
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        kernel_size = net.get_shape()[1:3]
        if kernel_size.is_fully_defined():
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_8x8')
        else:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
        if not num_classes:
          return net, end_points
        net = slim.flatten(net)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')
        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points
flownet_c.default_image_size = 299

def flownet_c_vgg(input_a,
           input_b,
           num_classes=9,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='flownet_c',
           fc_conv_padding='VALID',
           global_pool=False):

  with tf.variable_scope(scope, 'flownet_c', [input_a, input_b]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      #conv_a = slim.repeat(input_a, 3, slim.conv2d, 64, [3,3], scope = 'conv_a')
      #conv_b = slim.repeat(input_b, 3, slim.conv2d, 64, [3,3], scope = 'conv_b')
      #cc = correlation(conv_a, conv_b, 1, 20, 1, 2, 20)
      net = slim.repeat(input_a, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
flownet_c_vgg.default_image_size = 224
