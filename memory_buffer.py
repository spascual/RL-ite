from typing import Dict, Optional

import numpy as np
from gym import Env

from trajectory import Step, Trajectory


class MemoryBuffer:
    """
    Implementation of a transition memory buffer
    """

    def __init__(self, max_memory_size, environment: Env, metadata: Optional[Dict] = None):

        dim_state = environment.observation_space.shape[0]
        dim_action = environment.action_space.shape[0]

        self.shape_dtype_dict = dict(
            state=(dim_state, np.float32),
            action=(dim_action, np.float32),
            reward=(1, np.float32),
            next_state=(dim_state, np.float32),
            termination_masks=(1, 'uint8')
        )
        metadata = {} if metadata is None else metadata
        self.shape_dtype_dict.update(metadata)
        self.max_memory_size = max_memory_size
        self.statistics = self.shape_dtype_dict.keys()

        self._memory_buffer = {
            stat_name: np.zeros(shape=(max_memory_size, self.shape_dtype_dict[stat_name][0]),
                                dtype=self.shape_dtype_dict[stat_name][1])
            for stat_name in self.statistics
        }
        self._top = 0  # Trailing index with latest entry in env
        self._size = 0  # Trailing index with num samplea

    def add_step(self, step: Step):
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
