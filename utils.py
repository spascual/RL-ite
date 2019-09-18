import time
from typing import TypeVar, Dict

import numpy as np
import tensorflow as tf
from gym import Env

State = TypeVar("State", tf.Tensor, np.ndarray)
Observation = TypeVar("Observation", tf.Tensor, np.ndarray, object)
Action = TypeVar("Action", tf.Tensor, np.ndarray)
Reward = TypeVar("Reward", tf.Tensor, np.ndarray)
StateAction = TypeVar("StateAction", tf.Tensor, np.ndarray)


def log_probability_gaussian(x, mu, std):
    """
    Computes the log-density of a diagonal multivariate normal.
    """
    d = x - mu
    alpha = d / std
    num_dims = tf.cast(d.shape[-1], std.dtype)
    p = -0.5 * tf.reduce_sum(tf.square(alpha), axis=-1)
    p -= 0.5 * num_dims * np.log(2 * np.pi)
    p -= tf.reduce_sum(tf.math.log(std), axis=-1)
    return tf.reshape(p, (-1, 1))


def metadata_to_numpy(metadata: Dict):
    return {
        key: value.numpy() if isinstance(value, tf.Tensor) else value
        for key, value in metadata.items()
    }


def generate_experiment_signature(env: Env):
    time_signature = time.strftime("%Y-%m-%d_%H-%M")
    env_name = env.__class__.__name__
    return f"/tmp/rl-ite/{env_name}/{time_signature}"
