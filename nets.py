from copy import deepcopy
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from gym import Env

from config import BasicPolicyConfigSAC, BasicLearnerConfigSAC
from utils import Action, State, log_probability_gaussian, StateAction

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
            # assert not np.all(np.isnan(action.numpy())), nans_message
            log_pi = log_probability_gaussian(action, mu, std)
            metadata = dict(log_pi=log_pi,
                            mean=mu,
                            std=std)
            return action, metadata
        return action, {}

    def get_action_mean(self, state: State):
        mu, _ = self.model(state)
        return mu, {}

    def __call__(self, state: State, is_learning: bool):
        if is_learning:
            return self.sample_action(state, include_metadata=True)
        else:
            return self.get_action_mean(state)


class BoundedGaussianPolicy(GaussianPolicy):
    def get_action_mean(self, state: State):
        action_unconst_mean, _ = super().get_action_mean(state)
        metadata = dict(action_unconst=action_unconst_mean)
        return tf.nn.tanh(action_unconst_mean), metadata

    def sample_action(self, state: State, include_metadata: bool = False):
        action_unconst, metadata = super().sample_action(state, include_metadata=include_metadata)
        action = tf.nn.tanh(action_unconst)
        metadata.update(dict(action_unconst=action_unconst))
        return action, metadata


class QNetworkSAC(tf.Module):
    """
    Q-value function approximator.
    """

    def __init__(self, environment: Env, config: BasicLearnerConfigSAC):
        super().__init__()
        dim_state = environment.observation_space.shape[0]
        dim_action = environment.action_space.shape[0]
        self.dim_input = dim_state + dim_action

        self._qnet_1, self._qnet_2 = self.create_models(config)

    def create_models(self, config: BasicLearnerConfigSAC) -> Tuple[keras.Model, keras.Model]:
        model_1 = self.create_single_model(config)
        model_2 = self.create_single_model(config)
        return model_1, model_2

    def create_single_model(self, config: BasicLearnerConfigSAC) -> keras.Model:
        input_state = keras.Input(shape=(self.dim_input,), name='state')
        layers = [
            keras.layers.Dense(config.Qhidden_units, activation=config.Qactivation)
            for _ in range(config.Qhidden_layers)
        ]
        block = keras.Sequential(layers)(input_state)
        output = keras.layers.Dense(1)(block)
        model = keras.Model(input_state, output)

        # Compiple model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=config.learning_rate_Q),
            loss=tf.keras.metrics.MSE
        )
        return model

    @property
    def qnet_1(self):
        return self._qnet_1

    @property
    def qnet_2(self):
        return self._qnet_2

    def __call__(self, state_action: StateAction) -> tf.Tensor:
        return tf.minimum(self._qnet_1(state_action), self._qnet_2(state_action))


class VNetworkSAC(tf.Module):
    """
    V-value function approximator.
    """

    def __init__(self, environment: Env, config: BasicLearnerConfigSAC):
        super().__init__()
        self.dim_input = environment.observation_space.shape[0]
        self.tau = config.tau_V

        self._vnet, self._targetnet = self.create_models(config)

    def create_models(self, config: BasicLearnerConfigSAC) -> Tuple[keras.Model, keras.Model]:
        model_1 = self.create_single_model(config)
        model_2 = self.create_single_model(config)
        return model_1, model_2

    def create_single_model(self, config: BasicLearnerConfigSAC) -> keras.Model:
        input_state = keras.Input(shape=(self.dim_input,), name='state')
        layers = [
            keras.layers.Dense(config.Vhidden_units, activation=config.Vactivation)
            for _ in range(config.Vhidden_layers)
        ]
        block = keras.Sequential(layers)(input_state)
        output = keras.layers.Dense(1)(block)
        model = keras.Model(input_state, output)

        # Compiple model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=config.learning_rate_V),
            loss=tf.keras.metrics.MSE
        )
        return model

    @property
    def vnet(self):
        return self._vnet

    @property
    def targetnet(self):
        return self._targetnet

    def update_target(self):
        for target_var, var in zip(self._targetnet.trainable_variables,
                                   self._vnet.trainable_variables):
            target_var.assign(
                target_var * (1. - self.tau) + var * self.tau
            )
