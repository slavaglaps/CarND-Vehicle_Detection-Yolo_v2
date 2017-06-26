import numpy as np
import cv2
from keras import backend as K
import keras # broken for keras >= 2.0, use 1.2.2
from keras.models import Sequential
from keras.layers.convolutional import  MaxPooling2D,Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Lambda
from keras.layers import Input
from keras.layers.merge import concatenate
import tensorflow as tf

def darknet19(inputs):
    x = Conv2D(32, (3, 3) ,padding='same', kernel_regularizer= l2(5e-4), use_bias = False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='valid')(x)

    x = Conv2D(64,(3, 3) ,padding='same',use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='valid')(x)

    x = Conv2D(128,(3, 3),padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64,(1, 1) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (3, 3),padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='valid')(x)

    x = Conv2D(256,(3, 3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128,(1,1) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (3, 3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='valid')(x)

    x = Conv2D(512,(3, 3) ,padding='same',kernel_regularizer=l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256,(1, 1) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512,(3, 3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256,(1, 1) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512,(3, 3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='valid')(x)

    x = Conv2D(1024,(3, 3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512,(1, 1) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024,(3, 3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512,(1, 1) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(1024,(3, 3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return Model(inputs, x)

def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.
    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])

def yolo_body(inputs):
    Darknet19 = darknet19(inputs)
    conv20 = Conv2D(1024,(3,3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(Darknet19.output)
    conv20 = BatchNormalization()(conv20)
    conv20 = LeakyReLU(alpha=0.1)(conv20)

    conv20 = Conv2D(1024,(3,3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(conv20)
    conv20 = BatchNormalization()(conv20)
    conv20 = LeakyReLU(alpha=0.1)(conv20)

    conv13 = Darknet19.layers[42].output

    conv21 = Conv2D(64,(1,1) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(conv13)
    conv21 = BatchNormalization()(conv21)
    conv21 = LeakyReLU(alpha=0.1)(conv21)

    conv21_reshaped = Lambda(
            space_to_depth_x2,
            output_shape=space_to_depth_x2_output_shape,
            name='space_to_depth')(conv21)

    x = concatenate([conv21_reshaped, conv20])

    x = Conv2D(1024,(3,3) ,padding='same',kernel_regularizer= l2(5e-4),use_bias = False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(125,(1,1) ,padding='same',kernel_regularizer= l2(5e-4))(x)

    return  Model(inputs, x)

def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.
    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.
    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])

    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])

    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    # Static generation of conv_index:
    # conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    # conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    # conv_index = K.variable(
    #     conv_index.reshape(1, conv_height, conv_width, 1, 2))
    # feats = Reshape(
    #     (conv_dims[0], conv_dims[1], num_anchors, num_classes + 5))(feats)

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])
def yolo(inputs, anchors, num_classes):
    """Generate a complete YOLO_v2 localization model."""
    num_anchors = len(anchors)
    body = yolo_body(inputs, num_anchors, num_classes)
    outputs = yolo_head(body.output, anchors, num_classes)
    return outputs


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')

    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes

def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k
            
def draw_boxes(image, boxes, score, classes):  
    #image = cv2.imread(image)
    vocab_to_int = {c: i for i, c in enumerate(classes)}
    
    i=0
    for box in boxes:
        x1 = int(box[1])
        y1 = int(box[0])
        x2 = int(box[3])
        y2 = int(box[2])
        class_name = get_key(vocab_to_int,classes[i])
        cv2.rectangle(image,(x1,y1),( x2,y2),(0,255,0),4)
        font = cv2.FONT_HERSHEY_SIMPLEX     
        cv2.putText(image,class_name,(x1,y1-10), font, 1,(255,255,255),2)
        i+=1
    return image   