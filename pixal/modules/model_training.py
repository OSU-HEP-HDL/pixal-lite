import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def masked_mse_loss(y_true, y_pred):
    # Create a mask for non-zero values
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)

    # Compute squared error only where y_true is non-zero
    squared_error = tf.square(y_true - y_pred) * mask

    # Normalize by the number of valid (non-zero) entries
    return tf.reduce_sum(squared_error) / tf.reduce_sum(mask)

def resolve_loss(name):
    lookup = {
    'masked_mse_loss': masked_mse_loss,
    'mse': 'mse',
    'mae': 'mae'
        }
    return lookup.get(name, 'mse')
