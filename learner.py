from abc import abstractmethod
from typing import Optional

from gym import Env
import tensorflow as tf

from agent import Agent
from monitoring import Monitoring


class LearnerConfig(object):
    pass


class Learner(tf.Module):
    """Base class for learner that updates agent"""

    def __init__(self,
                 config: LearnerConfig,
                 enviroment: Env,
                 agent: Agent,
                 monitoring: Optional[Monitoring] = None,
                 **kwargs):
        super().__init__()
        self.config = config
        self.environment = enviroment
        self.agent = agent
        self.monitoring = monitoring

    @abstractmethod
    def learn_from_batch(self, batch):
        """Could return an agent or an agent checkpoint that can be loaded in a queue"""
        pass


class EmptyLearner(Learner):
    def __init__(self,
                 config: LearnerConfig,
                 enviroment: Env,
                 agent: Agent):
        super().__init__(config, enviroment, agent, monitoring=None)

    def learn_from_batch(self, batch):
        print('Agent is not learning')
        pass