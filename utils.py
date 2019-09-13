from typing import TypeVar, Dict
import tensorflow as tf
import numpy as np

State = TypeVar("State", tf.Tensor, np.ndarray)
Observation = TypeVar("Observation", tf.Tensor, np.ndarray, object)
Action = TypeVar("Action", tf.Tensor, np.ndarray)
Reward = TypeVar("Reward", tf.Tensor, np.ndarray)


def log_probability_gaussian(x, mu, L):
    """
    Kindly borrowed from GPflow :)

    Computes the log-density of a multivariate normal.
    :param x  : Dx1 or DxN sample(s) for which we want the density
    :param mu : Dx1 or DxN mean(s) of the normal distribution
    :param L  : DxD Cholesky decomposition of the covariance matrix
    :return p : (1,) or (N,) vector of log densities for each of the N x's and/or mu's
    x and mu are either vectors or matrices. If both are vectors (N,1):
    p[0] = log pdf(x) where x ~ N(mu, LL^T)
    If at least one is a matrix, we assume independence over the *columns*:
    the number of rows must match the size of L. Broadcasting behaviour:
    p[n] = log pdf of:
    x[n] ~ N(mu, LL^T) or x ~ N(mu[n], LL^T) or x[n] ~ N(mu[n], LL^T)
    """

    d = tf.reshape(x - mu, (1, -1, 1))
    if len(L.shape) == 2:
        L = tf.linalg.diag(L)

    alpha = tf.linalg.triangular_solve(L, d, lower=True)
    num_dims = tf.cast(d.shape[-2], L.dtype)
    p = -0.5 * tf.reduce_sum(tf.square(alpha), -2)
    p -= 0.5 * num_dims * np.log(2 * np.pi)
    p -= tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
    return p


def metadata_to_numpy(metadata: Dict):
    return {
        key: value.numpy() if isinstance(value, tf.Tensor) else value
        for key, value in metadata.items()
    }
