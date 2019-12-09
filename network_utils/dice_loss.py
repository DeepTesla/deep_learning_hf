import tensorflow as tf

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(0,1,2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(0,1,2))

    return 1 - numerator / denominator