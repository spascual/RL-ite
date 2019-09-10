from abc import abstractmethod
from typing import Tuple, Dict

import tensorflow as tf
import numpy as np
from gym import Env

from nets import BoundedGaussianPolicy
from utils import Action, State


class Agent(tf.Module):
    """
    Base class for an decision-making agent.
    """

    @abstractmethod
    def __init__(self, environment, **kwargs):
        pass

    @abstractmethod
    def load_policy(self, path: str):
        pass

    @abstractmethod
    def act(self, state: State) -> Tuple[Action, Dict]:
        pass

    @property
    def policy(self):
        return self._policy


class RandomAgent(Agent):
    """
    Random agent that samples actions from action space.
    """

    def __init__(self, environment: Env):
        super().__init__(environment)
        self.action_space = environment.action_space
        self.is_learning = True
        self._policy = None

    def act(self, state: State) -> Tuple[Action, Dict]:
        return tf.cast(self.action_space.sample(), tf.float32), {'log_pi': np.zeros(1, )}


class AgentSAC(Agent):
    """
    Agent for Soft Actor Critic algorithm.
    """

    def __init__(self, environment: Env, config: Basic):
        super().__init__(environment)
        self.action_space = environment.action_space
        self.is_learning = True
        self._policy = BoundedGaussianPolicy()

    def act(self, state: State) -> Tuple[Action, Dict]:
        return tf.cast(self.action_space.sample(), tf.float32), {'log_pi': np.zeros(1, )}
