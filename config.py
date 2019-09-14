from dataclasses import dataclass
from typing import Optional

@dataclass
class LearnerConfig:
    pass

@dataclass
class PolicyConfig:
    pass

@dataclass
class BasicLearnerConfigSAC(LearnerConfig):
    batch_size: Optional[int] = 256
    learning_rate_policy: Optional[float] = 3e-4
    learning_rate_Q: Optional[float] = 3e-4
    learning_rate_V: Optional[float] = 3e-4
    tau_V: Optional[float] = 0.005
    discount_factor: Optional[float] = 0.99
    alpha: Optional[float] = 50.0
    # Twin Q-value network config
    Qhidden_units: Optional[int] = 100
    Qhidden_layers: Optional[int] = 2
    Qactivation: Optional[str] = 'relu'
    # V-value and target network config
    Vhidden_units: Optional[int] = 100
    Vhidden_layers: Optional[int] = 2
    Vactivation: Optional[str] = 'relu'


@dataclass
class BasicPolicyConfigSAC:
    hidden_units: Optional[int] = 100
    hidden_layers: Optional[int] = 2
    activation: Optional[str] = 'relu'
    transform_std = None


@dataclass
class BasicConfigSAC:
    learner: BasicLearnerConfigSAC = BasicLearnerConfigSAC()
    policy: BasicPolicyConfigSAC = BasicPolicyConfigSAC()
    episode_horizon: Optional[int] = 100
    steps_before_learn: Optional[int] = 10
    memory_size: Optional[int] = int(3e6)
