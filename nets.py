import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from gym import Env

from config import BasicPolicyConfigSAC
from utils import Action, State, log_probability_gaussian

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianPolicy(tf.Module):
    def __init__(self, environment: Env, config: BasicPolicyConfigSAC):
        super().__init__()
        self.is_not_deterministic = False
        self.dim_state = environment.observation_space.shape[0]
        self.dim_action = environment.action_space.shape[0]
        self.transform_std = tf.math.exp if config.transform_std is None else config.transform_std

        self.model = self.create_model(config)
        self._normal = tfp.distributions.Normal(loc=0., scale=1.)

    def create_model(self, config: BasicPolicyConfigSAC) -> keras.Model:
        input_state = keras.Input(shape=(self.dim_state,), name='state')
        layers = [
            keras.layers.Dense(config.hidden_units, activation=config.activation)
            for _ in range(config.hidden_layers)
        ]
        policy_block = keras.Sequential(layers)(input_state)
        mean = keras.layers.Dense(self.dim_action)(policy_block)
        log_std = keras.layers.Dense(self.dim_action)(policy_block)
        std = self.transform_std(tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return keras.Model(inputs=input_state, outputs=(mean, std))

    def sample_action(self, state: State, include_metadata: bool = False):
        mu, std = self.model(state)
        action = mu + self._normal.sample(mu.shape) * std
        if include_metadata:
            nans_message = f"Actions return by agent policy contains NaNs"
            assert not tf.reduce_all(tf.math.is_nan(action)), nans_message
            log_pi = log_probability_gaussian(action, mu, std)
            metadata = dict(log_pi=log_pi,
                            mean=mu,
                            std=std)
            return action, metadata
        return action, {}

    def get_action_mean(self, state: State):
        mu, _ = self.model(state)
        return

    def __call__(self, state: State, is_learning: bool):
        if is_learning:
            return self.sample_action(state, include_metadata=True)
        else:
            return self.get_action_mean(state)


class BoundedGaussianPolicy(GaussianPolicy):
    def get_action_mean(self, state: State):
        action_unconst_mean = super().get_action_mean(state)
        return tf.nn.tanh(action_unconst_mean)

    def sample_action(self, state: State, include_metadata: bool = False):
        action_unconst, metadata = super().sample_action(state, include_metadata=include_metadata)
        action = tf.nn.tanh(action_unconst)
        return action, metadata



class QFunctionSAC(tf.Module):
    """
    Both Q and V are trained wrt to some mean squared error, so we can trained them with keras
    """
    pass


class VFunctionSAC(tf.Module):
    pass
