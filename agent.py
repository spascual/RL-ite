from abc import abstractmethod
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf
from gym import Env

from config import BasicPolicyConfigSAC
from nets import BoundedGaussianPolicy
from utils import Action, State, metadata_to_numpy, Observation


class Agent(tf.Module):
    """
    Base class for an decision-making agent.
    """

    def __init__(self, environment, **kwargs):
        super().__init__()
        self.environment = environment

    @abstractmethod
    def load_policy(self, path: str):
        pass

    @abstractmethod
    def act(self, state: State) -> Tuple[Action, Dict]:
        pass


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
        action = tf.cast(self.action_space.sample(), tf.float32)
        metadata = dict(log_pi=np.zeros(1, ))
        return action, metadata


class AgentSAC(Agent):
    """
    Agent for Soft Actor Critic algorithm.
    """

    def __init__(self, environment: Env, config: BasicPolicyConfigSAC):
        super().__init__(environment)
        self.action_space = environment.action_space
        self.is_learning = True
        self.policy = BoundedGaussianPolicy(environment, config)

    def act(self, observation: Observation) -> Tuple[Action, Dict]:
        state = np.atleast_2d(observation)
        action, metadata = self.policy(state, is_learning=self.is_learning)
        action = action.numpy().reshape(-1)
        return action, metadata_to_numpy(metadata)
