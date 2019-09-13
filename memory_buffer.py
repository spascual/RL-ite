from typing import Dict, Optional

import numpy as np
from gym import Env

from trajectory import Step, Trajectory


class MemoryBuffer:
    """
    Implementation of a transition memory buffer
    """

    def __init__(self, max_memory_size):
        self.max_memory_size = max_memory_size

        self._top = 0  # Trailing index with latest entry in env
        self._size = 0  # Trailing index with num samplea

    def create_memory_buffer(self, step: Step):
        self.shape_dtype_dict = {
            stat_name: (value.shape[-1], value.dtype)
            for stat_name, value in step.asdict().items()
        }
        self.statistics = self.shape_dtype_dict.keys()

        self._memory_buffer = {
            stat_name: np.zeros(shape=(self.max_memory_size, self.shape_dtype_dict[stat_name][0]),
                                dtype=self.shape_dtype_dict[stat_name][1])
            for stat_name in self.statistics
        }

    def add_step(self, step: Step):
        if self._size == 0:
            self.create_memory_buffer(step)
        error_msg = f"""Memory buffer and step have different statistics - step:{step.statistics}, 
                    buffer:{self.statistics}"""
        assert step.statistics == self.statistics, error_msg
        for key, value in step.asdict().items():
            self._memory_buffer[key][self._top] = value
        self._advance()

    def add_transitions(self, trajectory: Trajectory):
        error_msg = f"""Memory buffer and trajectory have different statistics - 
                    traj:{trajectory.step_statistics}, buffer:{self.statistics}"""
        assert trajectory.step_statistics == self.statistics, error_msg
        for step in trajectory.step_history:
            self.add_step(step)

    def _advance(self):
        self._top = (self._top + 1) % self.max_memory_size
        if self._size < self.max_memory_size:
            self._size += 1

    def sample_batch_transitions(self, batch_size):
        assert batch_size <= self._size, "Not enough samples in buffer"
        indices = np.random.randint(0, self._size, batch_size)
        batch = {
            stat: self._memory_buffer[stat][indices] for stat in self.statistics
        }
        return batch

    @property
    def current_size(self):
        return self._size
