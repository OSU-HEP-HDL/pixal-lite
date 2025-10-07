import tensorflow as tf
from tensorflow import initializers
from keras.saving import register_keras_serializable
from tensorflow.keras.losses import Huber
import numpy as np


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
    'mae': 'mae',
    'huber': Huber(delta=1.0)
        }
    return lookup.get(name, 'mse')


def _reduce_mse(err):  # err: (..., )
    return tf.reduce_mean(tf.square(err))

def _reduce_mae(err):
    return tf.reduce_mean(tf.abs(err))

def _reduce_rmse(err):
    return tf.sqrt(tf.reduce_mean(tf.square(err)) + 1e-12)

def make_channel_metric(idx: int, name: str, reducer=_reduce_mse):
    """
    Returns a tf.keras metric function computing a per-channel reduction.
    idx: channel index in the last dimension
    name: metric name shown in logs
    reducer: one of _reduce_mse / _reduce_mae / _reduce_rmse or your own
    """
    def metric(y_true, y_pred):
        ch_true = y_true[..., idx]
        ch_pred = y_pred[..., idx]
        return reducer(ch_true - ch_pred)
    metric.__name__ = name  # important so Keras logs the right name
    return metric

# ---------- common reshape ----------
def _reshape_to_PC(t, C: int):
    """Reshape last dim to (..., P, C); works for last dim == C (P=1) or flattened P*C."""
    D = tf.shape(t)[-1]
    c = tf.constant(C, tf.int32)
    with tf.control_dependencies([
        tf.debugging.assert_equal(D % c, 0, message="Last dim must be divisible by num channels")
    ]):
        P = D // c
    new_shape = tf.concat([tf.shape(t)[:-1], tf.stack([P, c])], axis=0)
    return tf.reshape(t, new_shape), P

# ---------- weighted losses (works for flattened or channel-last) ----------
def make_weighted_loss(channel_names, weights, base="mse", delta=1.0, use_mask=False, from_logits=False):
    """
    Weighted loss supporting {mse, mae, huber, charbonnier, bce}.
    If use_mask=True, masks out elements where y_true == 0.
    """
    C = len(channel_names)
    if len(weights) != C:
        raise ValueError("weights list length must match channel_names length")
    w_vec = tf.constant(weights, tf.float32)  # (C,)

    def _elem_loss(diff, y_true=None, y_pred=None):
        if base == "mse":
            return tf.square(diff)
        if base == "mae":
            return tf.abs(diff)
        if base == "huber":
            abs_err = tf.abs(diff)
            quadratic = tf.minimum(abs_err, delta)
            linear = abs_err - quadratic
            return 0.5 * tf.square(quadratic) + delta * linear
        if base == "charbonnier":
            eps = 1e-3
            return tf.sqrt(tf.square(diff) + eps*eps) - eps
        if base == "bce":
            if from_logits:
                return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            else:
                eps = 1e-7
                y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
                return -(y_true*tf.math.log(y_pred) + (1-y_true)*tf.math.log(1-y_pred))
        raise ValueError("base must be in {'mse','mae','huber','charbonnier','bce'}")

    def loss(y_true, y_pred):
        if base == "bce":
            y_true_pc, P = _reshape_to_PC(y_true, C)
            y_pred_pc, _ = _reshape_to_PC(y_pred, C)
            per_elem = _elem_loss(None, y_true_pc, y_pred_pc)
        else:
            diff = y_true - y_pred
            diff_pc, P = _reshape_to_PC(diff, C)
            per_elem = _elem_loss(diff_pc, y_true, y_pred)

        per_elem = per_elem * w_vec  # apply channel weights

        if use_mask:
            # mask out y_true == 0
            mask = tf.cast(y_true > 0.0, tf.float32)
            mask_pc, _ = _reshape_to_PC(mask, C)
            per_elem = per_elem * mask_pc
            denom = tf.reduce_sum(mask_pc) + 1e-12
            return tf.reduce_sum(per_elem) / denom

        return tf.reduce_mean(per_elem)

    loss.__name__ = f"weighted_{base}"
    return loss




# ---------- per-channel metrics (handles flattened or channel-last) ----------
def make_per_channel_metric(idx: int, channel_names, reducer="mse"):
    """
    Logs per-channel error for arbitrary shapes. reducer in {'mse','mae','rmse'}.
    """
    C = len(channel_names)
    name = f"{reducer}_{channel_names[idx]}"

    def metric(y_true, y_pred):
        diff = y_true - y_pred
        diff_pc, _ = _reshape_to_PC(diff, C)            # (..., P, C)
        ch = diff_pc[..., idx]                          # (..., P)
        if reducer == "mse":
            val = tf.reduce_mean(tf.square(ch))
        elif reducer == "mae":
            val = tf.reduce_mean(tf.abs(ch))
        elif reducer == "rmse":
            val = tf.sqrt(tf.reduce_mean(tf.square(ch)) + 1e-12)
        else:
            raise ValueError("reducer must be in {'mse','mae','rmse'}")
        return val

    metric.__name__ = name
    return metric

# ---------- (optional) total weighted metric mirroring the loss ----------
def make_total_weighted_metric(channel_names, weights, base="mse", delta=1.0):
    C = len(channel_names)
    w_vec = tf.constant(weights, tf.float32)

    def metric(y_true, y_pred):
        diff = y_true - y_pred
        diff_pc, _ = _reshape_to_PC(diff, C)
        if base == "mse":
            per_elem = tf.square(diff_pc)
        elif base == "mae":
            per_elem = tf.abs(diff_pc)
        elif base == "huber":
            abs_err = tf.abs(diff_pc)
            quadratic = tf.minimum(abs_err, delta)
            linear = abs_err - quadratic
            per_elem = 0.5 * tf.square(quadratic) + delta * linear
        elif base == "charbonnier":
            eps = 1e-3
            per_elem = tf.sqrt(tf.square(diff_pc) + eps*eps) - eps
        else:
            raise ValueError
        per_elem = per_elem * w_vec
        return tf.reduce_mean(per_elem)

    metric.__name__ = f"weighted_{base}_metric"
    return metric

def make_weighted_contrib_metric(idx, channel_names, weights, base="huber", delta=0.5):
    C = len(channel_names)
    w = tf.constant(weights, tf.float32)[idx]

    def metric(y_true, y_pred):
        # reshape (..., P, C)
        D = tf.shape(y_true)[-1]
        P = D // C
        y_t = tf.reshape(y_true, tf.concat([tf.shape(y_true)[:-1], [P, C]], 0))
        y_p = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:-1], [P, C]], 0))
        d = y_t - y_p
        if base == "huber":
            ae = tf.abs(d[..., idx])
            q = tf.minimum(ae, delta)
            l = ae - q
            per_elem = 0.5*tf.square(q) + delta*l
        else:
            per_elem = tf.square(d[..., idx])  # MSE fallback
        return tf.reduce_mean(w * per_elem)

    metric.__name__ = f"wcontrib_{channel_names[idx]}"
    return metric


def make_channel_last_metric(idx, name, reducer=_reduce_mse):
    def metric(y_true, y_pred):
        return reducer(y_true[..., idx] - y_pred[..., idx])
    metric.__name__ = name
    return metric

def make_flat_channel_metric(idx, num_channels, name, reducer=_reduce_mse):
    def metric(y_true, y_pred):
        diff = y_true - y_pred                       # (..., D=P*C)
        D = tf.shape(diff)[-1]
        C = tf.constant(num_channels, tf.int32)
        with tf.control_dependencies([tf.debugging.assert_equal(D % C, 0)]):
            P = D // C
        new_shape = tf.concat([tf.shape(diff)[:-1], tf.stack([P, C])], axis=0)
        diff_pc = tf.reshape(diff, new_shape)        # (..., P, C)
        ch = diff_pc[..., idx]                       # (..., P)
        return reducer(ch)
    metric.__name__ = name
    return metric

def build_channel_metrics(channel_names, flattened=False, reducers=("mse",)):
    name_to_reducer = {"mse": _reduce_mse, "mae": _reduce_mae, "rmse": _reduce_rmse}
    C = len(channel_names)
    mets = []
    for i, ch in enumerate(channel_names):
        for r in reducers:
            reducer = name_to_reducer[r]
            name = f"{r}_{ch}"
            if flattened:
                mets.append(make_flat_channel_metric(i, C, name, reducer))
            else:
                mets.append(make_channel_last_metric(i, name, reducer))
    return mets

def compute_channel_means(x_train: np.ndarray, C: int) -> np.ndarray:
    """
    x_train: shape (N, D) flattened, where D = P*C (P pixels/features per channel)
             or shape (N, H, W, C). Returns channel means in [0,1], shape (C,).
    """
    if x_train.ndim == 2:
        N, D = x_train.shape
        assert D % C == 0, f"D={D} must be divisible by C={C}"
        P = D // C
        means = []
        for c in range(C):
            sl = slice(c * P, (c + 1) * P)
            means.append(float(x_train[:, sl].mean()))
        return np.array(means, dtype=np.float32)
    elif x_train.ndim == 4:
        # (N, H, W, C)
        return x_train.mean(axis=(0, 1, 2)).astype(np.float32)
    else:
        raise ValueError(f"Unexpected x_train shape {x_train.shape}")

def _to_logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Elementwise inverse-sigmoid; clamps away from 0/1 for numerical safety."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float32)

def make_flat_output_bias(input_dim: int, channel_means: np.ndarray) -> np.ndarray:
    """
    For flattened output (D = P*C), build bias vec of length D:
    [logit(m0)] repeated P times, then [logit(m1)] repeated P times, etc.
    """
    C = channel_means.shape[0]
    assert input_dim % C == 0, f"input_dim={input_dim} must be divisible by C={C}"
    P = input_dim // C
    logits = _to_logit(channel_means)          # shape (C,)
    blocks = [np.full(P, logits[c], dtype=np.float32) for c in range(C)]
    return np.concatenate(blocks, axis=0)     # shape (D,)

def build_flat_bias_initializer(
    name: str,
    input_dim: int,
    *,
    channel_means: np.ndarray | None = None,
    C: int | None = None,
    seed: int | None = None,
):
    """
    Returns a Keras bias initializer suitable for a Dense(..., bias_initializer=...).
    Produces a *flat vector* of length input_dim for common names:
      - 'none' -> None
      - 'zeros'/'zero' -> Constant(0) vector
      - 'ones' -> Constant(1) vector
      - 'constant:<v>' -> Constant(v) vector
      - 'channel_means' -> Constant(make_flat_output_bias(...))  # needs channel_means and C
      - 'random_normal[:mean,std]' -> RandomNormal (lets Keras allocate shape)
      - 'random_uniform[:min,max]' -> RandomUniform (lets Keras allocate shape)
    """
    if name is None:
        return None
    key = str(name).lower().strip()

    # 1) no bias
    if key == 'none':
        return None

    # 2) flat constants
    if key in ('zeros', 'zero'):
        return initializers.Constant(np.zeros((input_dim,), dtype=np.float32))
    if key == 'ones':
        return initializers.Constant(np.ones((input_dim,), dtype=np.float32))
    if key.startswith('constant:'):
        try:
            val = float(key.split(':', 1)[1])
        except Exception as e:
            raise ValueError(f"Could not parse constant value from '{name}'") from e
        return initializers.Constant(np.full((input_dim,), val, dtype=np.float32))

    # 3) channel-means → flat per-channel logit bias
    if key == 'channel_means':
        if channel_means is None or C is None:
            raise ValueError("channel_means and C are required for 'channel_means' bias")
        vec = make_flat_output_bias(input_dim, np.asarray(channel_means, dtype=np.float32))
        return initializers.Constant(vec)

    # 4) random distributions (let Keras create vector of correct shape)
    if key.startswith('random_normal'):
        # allow 'random_normal' or 'random_normal:mean,std'
        mean, std = 0.0, 0.05
        if ':' in key:
            try:
                mean, std = map(float, key.split(':', 1)[1].split(','))
            except Exception as e:
                raise ValueError(f"Use 'random_normal:mean,std' (got '{name}')") from e
        return initializers.RandomNormal(mean=mean, stddev=std, seed=seed)

    if key.startswith('random_uniform'):
        # allow 'random_uniform' or 'random_uniform:min,max'
        minv, maxv = -0.05, 0.05
        if ':' in key:
            try:
                minv, maxv = map(float, key.split(':', 1)[1].split(','))
            except Exception as e:
                raise ValueError(f"Use 'random_uniform:min,max' (got '{name}')") from e
        return initializers.RandomUniform(minval=minv, maxval=maxv, seed=seed)

    raise ValueError(f"Unknown bias initializer '{name}'")
