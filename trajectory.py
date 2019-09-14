from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import numpy as np

from utils import Observation, Action, Reward



class Trajectory:
    """List of steps recording Agent-Environment interactions over an episode"""

    def __init__(self):
        self.step_count = 0
        self.trajectory_returns: np.ndarray = np.zeros((1, 1), np.float32)
        self.trajectory_entropy: np.ndarray = np.zeros((1, 1), np.float32)


    def initialize_records(self, step: 'Step'):
        self._step_statistics = step.statistics
        self.step_history = [step]
        self.trajectory_buffer = {stat_name: [value] for stat_name, value in step.asdict().items()}

    def register_step(self,
                      observation: Observation,
                      action: Action,
                      reward: Reward,
                      next_observation: Observation,
                      done: bool,
                      action_metadata: dict) -> 'Step':

        self.trajectory_returns += reward
        if 'log_pi' in action_metadata.keys():
            self.trajectory_entropy += action_metadata['log_pi'].reshape(1, -1)

        action_metadata.update(dict(returns=self.trajectory_returns,
                                    entropy=self.trajectory_entropy))

        step = Step(state=observation.reshape(1, -1),
                    action=action.reshape(1, -1),
                    reward=np.array(reward, dtype=np.float32).reshape(1, -1),
                    next_state=next_observation.reshape(1, -1),
                    termination_masks=np.array(done, dtype='uint8').reshape(1, -1),
                    metadata=action_metadata
                    )
        if self.step_count == 0:
            self.initialize_records(step)
        else:
            for stat, value in step.asdict().items():
                self.trajectory_buffer[stat].append(value)
            self.step_history.append(step)

        self.step_count += 1
        return step

    def load_trajectory_buffer(self, step_history: List['Step']):
        self.trajectory_buffer = {
            stat: [step.asdict()[stat] for step in step_history]
            for stat in step_history[0].asdict().keys()
        }

    @property
    def step_statistics(self):
        return self._step_statistics

    def __len__(self):
        return len(self.step_history)

    def __getitem__(self, stat_name):
        return self.trajectory_buffer[stat_name]


@dataclass
class Step:
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    termination_masks: np.ndarray
    metadata: Optional[Dict] = None

    def asdict(self):
        stat_dict = asdict(self)
        if self.metadata is not None:
            stat_dict.update(stat_dict.pop('metadata'))
        return stat_dict

    @property
    def statistics(self):
        return set(self.asdict().keys())
