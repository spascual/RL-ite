import functools
from collections import Callable

import tensorflow as tf


class Monitoring:
    def __init__(self, log_path: str = '/tmp/learner-2'):
        self.log_path = log_path
        self.summary_writter = tf.summary.create_file_writer(log_path)

    @classmethod
    def record_loss(cls, name: str, step: int):
        def wrapper(closure):
            loss = closure()
            tf.summary.scalar(name, loss, step=step)
            return closure
        return wrapper
