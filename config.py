from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LearnerConfig:
    pass


@dataclass
class PolicyConfig:
    pass


@dataclass
class BasicLearnerConfigSAC(LearnerConfig):
    batch_size: Optional[int] = 256
    learning_rate_policy: Optional[float] = 1e-3
    learning_rate_Q: Optional[float] = 1e-3
    learning_rate_V: Optional[float] = 1e-3
    tau_V: Optional[float] = 0.005
    discount_factor: Optional[float] = 0.99
    alpha: Optional[float] = 50.
    # Twin Q-value network config
    Qhidden_units: Optional[Tuple[int]] = (128, 64)
    Qactivation: Optional[str] = 'relu'
    # V-value and target network config
    Vhidden_units: Optional[Tuple[int]] = (128, 64)
    Vactivation: Optional[str] = 'relu'


@dataclass
class BasicPolicyConfigSAC:
    hidden_units: Optional[Tuple[int]] = (128, 64)
    activation: Optional[str] = 'relu'
    transform_std = None


@dataclass
class BasicConfigSAC:
    learner: BasicLearnerConfigSAC = BasicLearnerConfigSAC()
    policy: BasicPolicyConfigSAC = BasicPolicyConfigSAC()
    episode_horizon: Optional[int] = 1000
    steps_before_learn: Optional[int] = 10
    memory_size: Optional[int] = int(3e6)
