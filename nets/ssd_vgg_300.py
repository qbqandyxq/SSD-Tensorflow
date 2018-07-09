import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common
import os
slim = tf.contrib.slim


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.
    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4be', 'block7be', 'block8be', 'block9be', 'block10be', 'block11'],
        #feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.1, 1.0],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        anchor_ratios=[[3.05, .55],
                       [4.2, .43, 2.1, .72],
                       [4.2, .43, 2.1, .72],
                       [4.2, .43, 2.1, .72],
                       [3.05, .55],
                       [3.05, .55],],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
        """SSD network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #
    def update_feature_shapes(self, predictions):
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = ssd_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            self.params.num_classes,
            self.params.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=self.params.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).
    This function follows the computation performed in the original
    implementation of SSD in Caffe.
    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.
    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.
    Determine the relative position grid of the centers, and the relative
    width and height.
    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.
    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    
    # Add first anchor boxes with ratio=1.
    
    '''
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    if sizes[0]==21:
        w[0]= 0.07#0.6226406042097197 ##
        h[0]= 0.07#0.06097823369670409 ##
        
        w[1]= 0.2817900581546291 
        h[1]= 0.1709254321034428
        
        w[2]= 0.12856059450050417
        h[2]= 0.5936804574323933
        
        w[3]= 0.1#0.6885893096118244 #0.1#
        h[3]= 0.1#0.11361447014366964 #0.1#
    elif sizes[0] == 45:
        w[0]= 0.5637220012089151
        h[0]= 0.2171666867125047
        
        w[1]= 0.37190687107777665 #0.15#
        h[1]= 0.33145800418837723 #0.15#
        
        w[2]= 0.28235313940395107
        h[2]= 0.5315628664971818
        
        w[3]= 0.8661445310667978
        h[3]= 0.17543346823316144
        
        w[4]= 0.2119578059491015
        h[4]= 0.8442917148587913
        
        w[5]= 0.6345084838163698
        h[5]= 0.34157773612074016
    elif sizes[0] == 99:
        w[0]= 0.879581810232616 #0.33#
        h[0]= 0.26326779228278535 #0.33#
        
        w[1]= 0.5182249773837754 
        h[1]= 0.453369415536223
        
        w[2]= 0.4400071011823397
        h[2]= 0.6208753147903942
        
        w[3]= 0.3249690423361021
        h[3]= 0.8506196694398569
        
        w[4]= 0.9361624310324689
        h[4]= 0.3613866153297065
        
        w[5]= 0.7570171535805835
        h[5]= 0.45007541305626997
        
    elif sizes[0] == 153:
        w[0]= 0.6323263151496324 #0.51#
        h[0]= 0.5625086520560226 #0.51#
        
        w[1]= 0.4286479134255227 
        h[1]= 0.896354799295622
        
        w[2]= 0.5639741779077256
        h[2]= 0.7099333873266612
        
        w[3]= 0.8127147069253777
        h[3]= 0.5911745420388036
        
        w[4]= 0.9669416662972017
        h[4]= 0.5006411236510211
        
        w[5]= 0.709857711696845
        h[5]= 0.7032392550257252
    
    elif sizes[0] == 207:
        w[0]= 0.5330277380605914 #0.69#
        h[0]= 0.9496621340603368 #0.69#
        
        w[1]= 0.634213359430225
        h[1]= 0.8798663440080869
        
        w[2]= 0.9676740197061093
        h[2]= 0.6636433675358576
        
        w[3]= 0.8212706899243332
        h[3]= 0.7826311634797078
    elif sizes[0] == 261:
        w[0]= 0.9722688197539527#0.87
        h[0]= 0.8200709629893037#0.87
        
        w[1]= 0.7290641364912129 
        h[1]= 0.9512706941293938
        
        w[2]= 0.8544768439969375
        h[2]= 0.9510762036216801
        
        w[3]= 0.9877004263211381
        h[3]= 0.9796718267824008
    print("================hhhhhhhh=================", h)
    print("================wwwwwwww=================", w)
    
    
    '''
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    print("================hhhhhhhh=================", h)
    print("================wwwwwwww=================", w)
    
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    #print("====loc_pred===", loc_pred.shape)
    loc_pred = custom_layers.channel_to_last(loc_pred)
    #print("====loc_pred===", loc_pred.shape)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    #print("====loc_pred===", loc_pred.shape)
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    #print("====cls_pred===", cls_pred.shape)
    cls_pred = custom_layers.channel_to_last(cls_pred)
    #print("====cls_pred===", cls_pred.shape)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    #print("====cls_pred===", cls_pred.shape)
    return cls_pred, loc_pred


def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """SSD net definition.
    """
    #if data_format == 'NCHW':
    
    #inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))
    print("====", inputs.shape)
    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        #original
        
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        #net4_ass = slim.repeat(net4, 3, slim.conv2d, 512, [1, 1], scope='conv4_ass')
        #net4 = net4 * net4_ass
        
        #print("============", net4.shape)
        
        # 38x38
        #end_points['block4'] = net
        net = slim.max_pool2d(net4, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        #end_points['block7'] = net
        net7 = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        
        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net7, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net8 = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        #end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net8, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net9 = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        #end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net9, 128, [1, 1], scope='conv1x1')
            net10 = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        #end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net11 = slim.conv2d(net10, 128, [1, 1], scope='conv1x1')
            net11 = slim.conv2d(net11, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net11
        #print("================net11 shape", net11.shape)
        # NCHW
        #################################################################################
        end_point = 'block10be'
        with tf.variable_scope(end_point):
            net10_a = tf.transpose(net11, perm=(0, 2, 3, 1))#nchw
            net10_a = tf.image.resize_nearest_neighbor(net10_a, (3,3))
            net10_a = tf.transpose(net10_a, perm=(0, 3, 1, 2))#nchw
            
            net10_a = slim.conv2d(net10_a, 256, 3, scope='pre10_3x3')
            net10_b = slim.conv2d(net10, 256, 1, scope='pre10_1x1')
            net10_o = net10_a * net10_b
            net10_o = slim.conv2d(net10_o, 256, 3, scope='pre10_3x3_')
        end_points[end_point] = net10_o #3
        
        end_point = 'block9be'
        with tf.variable_scope(end_point):
            net9_a = tf.transpose(net10_o, perm=(0, 2, 3, 1))#nchw
            net9_a = tf.image.resize_nearest_neighbor(net9_a, (5,5))
            net9_a = tf.transpose(net9_a, perm=(0, 3, 1, 2))#nchw
            
            net9_a = slim.conv2d(net9_a, 256, [3,3], scope='pre9_3x3')
            net9_b = slim.conv2d(net9, 256, [1, 1], scope='pre9_1x1')
            net9_o = net9_a * net9_b
            net9_o = slim.conv2d(net9_o, 256, 3, scope='pre9_3x3_')
        end_points[end_point] = net9_o#5
        
        end_point = 'block8be'
        with tf.variable_scope(end_point):
            net8_a = tf.transpose(net9_o, perm=(0, 2, 3, 1))#nchw
            net8_a = tf.image.resize_nearest_neighbor(net8_a, (10,10))
            net8_a = tf.transpose(net8_a, perm=(0, 3, 1, 2))#nchw
            
            net8_a = slim.conv2d(net8_a, 512, [3,3], padding='SAME', scope='pre8_3x3')
            net8_b = slim.conv2d(net8, 512, [1, 1], scope='pre8_1x1')#10
            net8_o = net8_a * net8_b
            net8_o = slim.conv2d(net8_o, 512, 3, scope='pre8_3x3_')
        end_points[end_point] = net8_o#10

        end_point = 'block7be'
        with tf.variable_scope(end_point):
            net7_a = tf.transpose(net8_o, perm=(0, 2, 3, 1))#nchw
            net7_a = tf.image.resize_nearest_neighbor(net7_a, (19, 19))#
            net7_a = tf.transpose(net7_a, perm=(0, 3, 1, 2))#nchw
            
            net7_a = slim.conv2d(net7_a, 1024, [3, 3], padding='SAME', scope='pre7_3x3')
            net7_b = slim.conv2d(net7, 1024, [1, 1], scope='pre7_1x1')
            net7_o = net7_a * net7_b
            net7_o = slim.conv2d(net7_o, 1024, 3, scope='pre7_3x3_')
        end_points[end_point] = net7_o
        
        end_point = 'block4be'
        with tf.variable_scope(end_point):
            net4_a = tf.transpose(net7_o, perm=(0, 2, 3, 1))#nchw
            net4_a = tf.image.resize_nearest_neighbor(net4_a, (38, 38))#
            net4_a = tf.transpose(net4_a, perm=(0, 3, 1, 2))#nchw
            
            net4_a = slim.conv2d(net4_a, 512, [3, 3], padding='SAME', scope='pre4_3x3')
            #net4_b = slim.conv2d(net4, 512, [1, 1], scope='pre4_1x1')
            net4_b = slim.conv2d(net4, 128, 1, padding='SAME', scope='pre4_1x1_1')# 128, 1
            net4_b = slim.conv2d(net4_b, 256, 3, padding='SAME', scope='pre4_1x1_2')# 256, 1
            net4_b = slim.conv2d(net4_b, 512, 1, padding='SAME', scope='pre4_1x1_3')# 512, 1
            
            #net4_ass = slim.repeat(net4, 3, slim.conv2d, 512, [1, 1], scope='conv4_ass')
            net4_o = net4_a * net4_b#38
            net4_o = slim.conv2d(net4_o, 512, 3, scope='pre4_3x3_')
        end_points[end_point] = net4_o
        '''
        # inputs = 300, 300, 3
        net = slim.repeat(inputs, 2, slim.conv2d, 32, [3, 3], scope='conv1')
        net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
        # 300,300, 64
        net1_1 = slim.max_pool2d(net, [2, 2], scope='pool1')# 150,150,64
        net1_2 = slim.conv2d(net, 96, [3, 3], stride=2, padding='SAME', scope='conv1_3') #150,150,96
        net1 = tf.concat(1, [net1_1, net1_2])# 150,150, 160
        
        with tf.variable_scope('Mixed_1a'):
            with tf.variable_scope('Branch_0'):
                net1_0_0 = slim.conv2d(net1, 64, [1,1], scope='conva_1')
                net1_0_1 = slim.conv2d(net1_0_0, 96, [3,3], scope='conva_2')
            with tf.variable_scope('Branch_1'):
                net1_1_0 = slim.conv2d(net1, 64, [1,1], scope='convb_1')
                net1_1_1 = slim.conv2d(net1_1_0, 96, 5, scope='convb_2')
            net1 = tf.concat(1, [net1_0_1, net1_1_1])#150, 150, 192
        
        
        
        '''
        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points
ssd_net.default_image_size = 300


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.
    Args:
      weight_decay: The l2 regularization coefficient.
    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.
    Args:
      caffe_scope: Caffe scope object with loaded weights.
    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)