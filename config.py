from dataclasses import dataclass
from typing import Optional


@dataclass
class BasicLearnerConfigSAC:
    batch_size: Optional[int] = 256
    learning_rate_policy: Optional[float] = 3e-4
    learning_rate_Q: Optional[float] = 3e-4
    learning_rate_V: Optional[float] = 3e-4
    tau_V: Optional[float] = 0.005
    discount_factor: Optional[float] = 0.99
    alpha: Optional[float] = 50.0


@dataclass
class BasicPolicyConfigSAC:
    hidden_units: Optional[int] = 100
    hidden_layers: Optional[int] = 2
    activation: Optional[str] = 'relu'


@dataclass
class BasicConfigSAC:
    learner: BasicLearnerConfigSAC = BasicLearnerConfigSAC()
    policy: BasicPolicyConfigSAC = BasicPolicyConfigSAC()
    episode_horizon: Optional[int] = 100
    steps_before_learn: Optional[int] = 10
    memory_size: Optional[int] = int(3e6)
