from abc import abstractmethod
from typing import Optional

from gym import Env
import tensorflow as tf

from agent import Agent, AgentSAC
from config import LearnerConfig, BasicLearnerConfigSAC
from monitoring import Monitoring
from nets import QNetworkSAC, VNetworkSAC


class Learner(tf.Module):
    """Base class for learner that updates agent"""

    def __init__(self,
                 config: LearnerConfig,
                 enviroment: Env,
                 agent: Agent,
                 monitoring: Optional[Monitoring] = None,
                 logging: Optional[bool] = True,
                 **kwargs):
        super().__init__()
        self.config = config
        self.environment = enviroment
        self.agent = agent
        self.monitoring = monitoring
        self.logging = logging
        self.learning_step = 0

    def _start_learning_step_callback(self):
        self.learning_step += 1
        if self.logging and self.learning_step % 100 == 0:
            print(f'Agent is learning, learning step #{self.learning_step}')

    @abstractmethod
    def learn_from_batch(self, batch):
        """Could return an agent or an agent checkpoint that can be loaded in a queue"""
        pass


class EmptyLearner(Learner):
    def __init__(self,
                 config: BasicLearnerConfigSAC,
                 enviroment: Env,
                 agent: Agent):
        super().__init__(config, enviroment, agent, monitoring=None)

    def learn_from_batch(self, batch):
        if self.logging:
            print('Agent is not learning')
        pass


class LearnerSAC(Learner):
    def __init__(self,
                 config: BasicLearnerConfigSAC,
                 enviroment: Env,
                 agent: AgentSAC,
                 monitoring: Optional[Monitoring] = None):
        super().__init__(config, enviroment, agent, monitoring)
        self.policy = agent.policy
        self.monitoring = monitoring

        self.gamma = config.discount_factor
        self.alpha = config.alpha

        self.q_nets = QNetworkSAC(enviroment, config)
        self.v_nets = VNetworkSAC(enviroment, config)
        self.policy_opt = tf.keras.optimizers.Adam(lr=config.learning_rate_policy)
        self.q_net_opt = tf.keras.optimizers.Adam(lr=config.learning_rate_Q)
        self.v_net_opt = tf.keras.optimizers.Adam(lr=config.learning_rate_V)

    @tf.function
    def learn_from_batch(self, batch):
        self._start_learning_step_callback()
        state = tf.cast(batch['state'], tf.float32)
        action = tf.cast(batch['action'], tf.float32)
        reward = self.alpha * batch['reward']
        next_state = tf.cast(batch['next_state'], tf.float32)
        termination_masks = tf.cast(batch['termination_masks'], tf.float32)
        state_action_pair = tf.concat([state, action], axis=-1)

        vnet, targetnet = self.v_nets.vnet, self.v_nets.targetnet
        # Target for Q-network MSE objective
        q_hat = reward + self.gamma * (1. - termination_masks) * targetnet(next_state)

        with tf.GradientTape(persistent=True) as tape:
            loss_policy, q_values_hat, log_pi_hat = self.policy_closure(state)
            loss_q_1, loss_q_2 = self.q_closure(state_action_pair, q_hat)
            v_hat = q_values_hat - log_pi_hat
            loss_v = self.v_closure(state, v_hat)
        grad_q_1 = tape.gradient(loss_q_1, self.q_nets.qnet_1.trainable_variables)
        grad_q_2 = tape.gradient(loss_q_2, self.q_nets.qnet_2.trainable_variables)
        grad_v = tape.gradient(loss_v, self.v_nets.vnet.trainable_variables)
        grad_pi = tape.gradient(loss_policy, self.policy.trainable_variables)

        self.q_net_opt.apply_gradients(zip(grad_q_1, self.q_nets.qnet_1.trainable_variables))
        self.q_net_opt.apply_gradients(zip(grad_q_2, self.q_nets.qnet_2.trainable_variables))
        self.v_net_opt.apply_gradients(zip(grad_v, self.v_nets.vnet.trainable_variables))
        self.policy_opt.apply_gradients(zip(grad_pi, self.policy.trainable_variables))
        self.v_nets.update_target()

    @tf.function
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
        tf.summary.scalar('loss_pi', loss_policy, step=self.learning_step)
        tf.summary.scalar('std', tf.reduce_mean(metadata['std']), step=self.learning_step)
        return loss_policy, q_values_hat, log_pi_hat


    @tf.function
    def q_closure(self, state_action_pair, q_hat):
        loss_q_1 = tf.metrics.MSE(self.q_nets.qnet_1(state_action_pair), q_hat)
        loss_q_2 = tf.metrics.MSE(self.q_nets.qnet_2(state_action_pair), q_hat)
        return loss_q_1, loss_q_2

    @tf.function
    def v_closure(self, state, v_hat):
        loss_v = tf.metrics.MSE(self.v_nets.vnet(state), v_hat)
        return loss_v
