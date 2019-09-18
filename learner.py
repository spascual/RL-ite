from abc import abstractmethod
from typing import Optional

import tensorflow as tf
from gym import Env

from agent import Agent, AgentSAC
from config import LearnerConfig, BasicLearnerConfigSAC
from monitor import Monitor
from nets import QNetworkSAC, VNetworkSAC


class Learner(tf.Module):
    """Base class for learner that updates agent"""

    def __init__(self,
                 config: LearnerConfig,
                 enviroment: Env,
                 agent: Agent,
                 monitor: Optional[Monitor] = None,
                 logging: Optional[bool] = True,
                 **kwargs):
        super().__init__()
        self.config = config
        self.environment = enviroment
        self.agent = agent
        self.monitor = monitor

    @abstractmethod
    def learn_from_batch(self, batch):
        """Could return an agent or an agent checkpoint that can be loaded in a queue"""
        pass


class EmptyLearner(Learner):
    def __init__(self,
                 config: BasicLearnerConfigSAC,
                 enviroment: Env,
                 agent: Agent):
        super().__init__(config, enviroment, agent, monitor=None)

    def learn_from_batch(self, batch):
        if self.logging:
            print('Agent is not learning')
        pass


class LearnerSAC(Learner):
    def __init__(self,
                 config: BasicLearnerConfigSAC,
                 enviroment: Env,
                 agent: AgentSAC,
                 monitor: Optional[Monitor] = None):
        super().__init__(config, enviroment, agent, monitor)
        self.policy = agent.policy
        self.monitor = monitor

        self.gamma = config.discount_factor
        self.alpha = config.alpha

        self.q_nets = QNetworkSAC(enviroment, config)
        self.v_nets = VNetworkSAC(enviroment, config)
        self.policy_opt = tf.keras.optimizers.Adam(lr=config.learning_rate_policy)
        self.q_net_opt = tf.keras.optimizers.Adam(lr=config.learning_rate_Q)
        self.v_net_opt = tf.keras.optimizers.Adam(lr=config.learning_rate_V)

    # @tf.function
    def learn_from_batch(self, batch):
        self.monitor.start_learning_step_callback()
        state = tf.cast(batch['state'], tf.float32)
        action = tf.cast(batch['action'], tf.float32)
        reward = self.alpha * batch['reward']
        next_state = tf.cast(batch['next_state'], tf.float32)
        termination_masks = tf.cast(batch['termination_masks'], tf.float32)
        state_action_pair = tf.concat([state, action], axis=-1)
        # Target for Q-network MSE objective
        targetnet = self.v_nets.targetnet
        q_hat = reward + self.gamma * (1. - termination_masks) * targetnet(next_state)

        with tf.GradientTape(persistent=True) as tape:
            loss_policy, q_values_hat, log_pi_hat = self.policy_closure(state)
            loss_q_0, loss_q_1 = self.q_closure(state_action_pair, q_hat)
            v_hat = q_values_hat - log_pi_hat
            loss_v = self.v_closure(state, v_hat)

        for loss_i, q_net_i in zip([loss_q_0, loss_q_1], self.q_nets):
            grad_q = tape.gradient(loss_i, q_net_i.trainable_variables)
            self.q_net_opt.apply_gradients(zip(grad_q, q_net_i.trainable_variables))

        grad_v = tape.gradient(loss_v, self.v_nets.vnet.trainable_variables)
        self.v_net_opt.apply_gradients(zip(grad_v, self.v_nets.vnet.trainable_variables))

        grad_pi = tape.gradient(loss_policy, self.policy.trainable_variables)
        self.policy_opt.apply_gradients(zip(grad_pi, self.policy.trainable_variables))

        self.v_nets.update_target()

    # @tf.function
    def policy_closure(self, state):
        action_hat, metadata = self.policy.sample_action(state, include_metadata=True)
        state_action_pair_hat = tf.concat([state, action_hat], axis=-1)
        mean_hat, std_hat = metadata['mean'], metadata['std']
        log_pi_hat = metadata['log_pi']

        q_values_hat = self.q_nets(state_action_pair_hat)

        # Loss for policy network using reparametrisation trick
        loss_policy = tf.reduce_mean(log_pi_hat - q_values_hat)
        loss_policy += 1e-3 * tf.reduce_mean(mean_hat ** 2)
        loss_policy += 1e-3 * tf.reduce_mean(tf.math.log(std_hat) ** 2)
        self.monitor.write_to_summary('loss_pi', loss_policy)
        self.monitor.write_to_summary('std', tf.reduce_mean(metadata['std']))
        return loss_policy, q_values_hat, log_pi_hat

    @tf.function
    def q_closure(self, state_action_pair, q_hat):
        loss_q_0, loss_q_1 = (
            tf.metrics.MSE(q_net(state_action_pair), q_hat) for q_net in self.q_nets
        )
        return loss_q_0, loss_q_1

    @tf.function
    def v_closure(self, state, v_hat):
        loss_v = tf.metrics.MSE(self.v_nets.vnet(state), v_hat)
        return loss_v
