from typing import TypeVar
import tensorflow as tf
import numpy as np

State = TypeVar("State", tf.Tensor, np.ndarray)
Observation = TypeVar("Observation", tf.Tensor, np.ndarray, object)
Action = TypeVar("Action", tf.Tensor, np.ndarray)
Reward = TypeVar("Reward", tf.Tensor, np.ndarray)

