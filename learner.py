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

        self.gamma = config.discount_factor
        self.q_nets = QNetworkSAC(enviroment, config)
        self.v_nets = VNetworkSAC(enviroment, config)
        self.policy_opt = tf.keras.optimizers.Adam(lr=config.learning_rate_policy)

    def learn_from_batch(self, batch):
        self._start_learning_step_callback()
        state = tf.cast(batch['state'], tf.float32)
        action = tf.cast(batch['action'], tf.float32)
        reward = batch['reward']
        next_state = tf.cast(batch['next_state'], tf.float32)
        termination_masks = batch['termination_masks']

        state_action_pair = tf.concat([state, action], axis=-1)

        qnet_1, qnet_2 = self.q_nets.qnet_1, self.q_nets.qnet_2
        vnet, targetnet = self.v_nets.vnet, self.v_nets.targetnet
        # Target for Q-network MSE objective
        q_hat = reward + self.gamma * (1. - termination_masks) * targetnet(next_state)

        action_hat, metadata = self.policy.sample_action(state, include_metadata=True)
        log_pi_hat = metadata['log_pi']
        state_action_pair_hat = tf.concat([state, action_hat], axis=-1)

        q_values_hat = self.q_nets(state_action_pair_hat)
        # Target for V-network MSE objective
        v_hat = q_values_hat.numpy() - log_pi_hat.numpy()

        # Fit Learner networks
        qnet_1.fit(state_action_pair, tf.cast(q_hat.numpy(), tf.float32),
                   verbose=0, epochs=1, steps_per_epoch=1)
        qnet_2.fit(state_action_pair, tf.cast(q_hat.numpy(), tf.float32),
                   verbose=0, epochs=1, steps_per_epoch=1)
        vnet.fit(state, tf.cast(v_hat, tf.float32),
                 verbose=0, epochs=1, steps_per_epoch=1)

        @tf.function
        def policy_closure():
            action_hat, metadata = self.policy.sample_action(state, include_metadata=True)
            mean_hat, std_hat = metadata['mean'], metadata['std']
            log_pi_hat = metadata['log_pi']
            # Loss for policy network using reparametrisation trick
            loss_policy = tf.reduce_mean(log_pi_hat - q_values_hat)
            loss_policy += 1e-3 * tf.reduce_mean(mean_hat ** 2)
            loss_policy += 1e-3 * tf.reduce_mean(tf.math.log(std_hat) ** 2)
            return loss_policy

        with tf.GradientTape() as tape:
            loss_policy = policy_closure()
            grads = tape.gradient(loss_policy, self.policy.trainable_variables)
        self.policy_opt.apply_gradients(zip(grads, self.policy.trainable_variables))
        self.v_nets.update_target()


        #
        # # Updates
        # if self.training_iteration % 50 == 0:
        #     shared_parameters = self.policy.model.parameters()
        #     theta = torch.cat([param.data.flatten() for param in shared_parameters])
        #     theta_min_max = [theta.min(), theta.max()]
        #     psi = torch.cat([param.data.flatten() for param in V_network.parameters()])
        #     psi_min_max = [psi.mean(), psi.mean() + 2 * psi.std(), psi.mean() - 2 * psi.std()]
        #     std_min_max = [std.min(), std.max()]
        #     self.monitor.record_data(self.training_iteration, 'reward',
        #                              (rewards / self.alpha).mean(),
        #                              'scalar')
        #     self.monitor.record_data(self.training_iteration, 'loss_policy', loss_policy, 'scalar')
        #     self.monitor.record_data(self.training_iteration, 'loss_Q', [loss_Q_0, loss_Q_1],
        #                              'vector')
        #     self.monitor.record_data(self.training_iteration, 'loss_V', loss_V, 'scalar')
        #     self.monitor.record_data(self.training_iteration, 'pi_param', theta_min_max, 'vector')
        #     self.monitor.record_data(self.training_iteration, 'V_param', psi_min_max, 'vector')
        #     self.monitor.record_data(self.training_iteration, 'std', std_min_max, 'vector')
        #
        # if self.training_iteration % 1000 == 0:
        #     # self.monitor.show_progress()
        #     self.save_results()
        #     self.policy.save_policy(self.save_path)
        #     pass

