import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_coef_logloss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_coef_multilabel(y_true, y_pred, M, smooth):
    dice = 0
    for index in range(M):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], smooth)
    return dice

def dice_coef_multilabel(y_true, y_pred, M, smooth_list):
    dice = 0
    for index in range(M):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], smooth_list[index])
    return dice

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

#def metric_dice():
#    def dice_coef(y_true, y_pred):
#        smooth = 1.
#        y_true_f = K.flatten(y_true)
#        y_pred_f = K.flatten(y_pred)
#        intersection = K.sum(y_true_f * y_pred_f)
#        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#        return dice
#    return dice_coef
#
#def metric_meanIoU(thres=0.4):
#    def metric(y_true,y_pred):
#        y_pred = tf.cast((y_pred > thres),tf.float32)
#        y_true_f = K.flatten(y_true)
#        y_pred_f = K.flatten(y_pred)
#        smooth=0
#        intersection = K.sum(y_true_f * y_pred_f)
#        union = K.sum(y_true_f) + K.sum(y_pred_f)  - intersection
#        mean_iou = K.mean((intersection + smooth) / (union + smooth))
#        return mean_iou
#    return metric


def recall(y_true, y_pred, c):
    #tf.print(y_true)
    pred_labels = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    true_labels = K.cast(K.argmax(y_true, axis=-1), K.floatx())
    #tf.print(K.argmax(y_true, axis=-1))
    #tf.print(K.argmax(y_pred, axis=-1))
    #tf.print(true_labels)
    #tf.print(pred_labels)
    #tf.print(true_labels == c)
    #tf.print(tf.logical_and(true_labels == c, pred_labels == c))
    #tf.print(K.cast(tf.logical_and(true_labels == c, pred_labels == c),K.floatx()))
    #tf.print(tf.reduce_sum(K.cast(tf.logical_and(true_labels == c, pred_labels == c),K.floatx())))
    #tf.print(type(tf.logical_and(true_labels == c, pred_labels == c)))

    tp = K.sum(K.cast(tf.logical_and(true_labels == c, pred_labels == c),K.floatx()))
    fn = K.sum(K.cast(tf.logical_and(true_labels == c, pred_labels != c),K.floatx()))
    tn = K.sum(K.cast(tf.logical_and(true_labels != c, pred_labels != c),K.floatx()))
    fp = K.sum(K.cast(tf.logical_and(true_labels != c, pred_labels == c),K.floatx()))

    sensitivity = (tp)/(tp+fn+0.0000001)
    return sensitivity


def precision(y_true, y_pred, c):
    pred_labels = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
    true_labels = K.cast(K.argmax(y_true, axis=-1), K.floatx())

    tp = K.sum(K.cast(tf.logical_and(true_labels == c, pred_labels == c),K.floatx()))
    fn = K.sum(K.cast(tf.logical_and(true_labels == c, pred_labels != c),K.floatx()))
    tn = K.sum(K.cast(tf.logical_and(true_labels != c, pred_labels != c),K.floatx()))
    fp = K.sum(K.cast(tf.logical_and(true_labels != c, pred_labels == c),K.floatx()))

    precision = (tp)/(tp+fp+0.0000001)
    return precision

def recall_c0(y_true, y_pred):
    return recall(y_true, y_pred, 0)

def precision_c0(y_true, y_pred):
    return precision(y_true, y_pred, 0)

def recall_c1(y_true, y_pred):
    return recall(y_true, y_pred, 1)

def precision_c1(y_true, y_pred):
    return precision(y_true, y_pred, 1)

def recall_c2(y_true, y_pred):
    return recall(y_true, y_pred, 2)

def precision_c2(y_true, y_pred):
    return precision(y_true, y_pred, 2)
