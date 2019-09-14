from typing import Tuple

import tensorflow as tf
from gym import Env

from gym.envs import box2d

import random
import time

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Dict

from agent import RandomAgent, AgentSAC
from config import BasicConfigSAC
from learner import EmptyLearner, LearnerSAC
from memory_buffer import MemoryBuffer, Trajectory
from monitoring import Monitoring
from utils import Observation

random.seed(42)

import gym


class TrainingLoopSAC:
    def __init__(self,
                 config: BasicConfigSAC,
                 environment: Env,
                 log_path=None
                 ):
        """
        TODO: Write docstring
        """
        self.config = config
        self.monitoring = Monitoring(log_path)
        self.env = environment

        self.batch_size = config.learner.batch_size
        self.episode_horizon = config.episode_horizon
        self.steps_before_learn = config.steps_before_learn

        self.memory_buffer = MemoryBuffer(max_memory_size=config.memory_size)

        # self.agent = RandomAgent(environment)
        # self.learner = EmptyLearner(
        #     config=config.learner,
        #     agent=self.agent,
        #     enviroment=self.env
        # )

        self.agent = AgentSAC(environment, config.policy)

        self.learner = LearnerSAC(
            config=config.learner,
            agent=self.agent,
            enviroment=self.env
        )

    def train(self,
              num_epochs: int = 30,
              steps_per_epoch: int = 1000,
              step_per_learning_step: int = 1,
              grad_updates_per_learning_step: int = 1):

        # assert self.agent.is_learning
        self.total_steps, self.learning_steps, self.episodes = 0, 0, 0

        observation, trajectory = self.reset_environment()
        for epoch in range(num_epochs):
            self.epoch_start_callback()
            for step in range(steps_per_epoch):
                observation, trajectory = self.register_agent_step(observation, trajectory)
                if step % step_per_learning_step == 0 and self.agent.is_learning:
                    self.learning_step(grad_updates_per_learning_step)
                    self.learning_steps += 1
            self.epoch_end_callback(steps_per_epoch)

        # plt.show(block=False)
        # plt.pause(20)
        # plt.close()

    def register_agent_step(self,
                            observation: Observation,
                            trajectory: Trajectory) -> Tuple[Observation, Trajectory]:
        action, action_metadata = self.agent.act(observation)
        next_observation, reward, done, info = self.env.step(action)
        # Append step to trajectory & add step to memory buffer
        step = trajectory.register_step(observation=observation,
                                        action=action,
                                        reward=reward,
                                        next_observation=next_observation,
                                        action_metadata=action_metadata,
                                        done=done)
        if self.agent.is_learning:
            self.memory_buffer.add_step(step)
        self.total_steps += 1
        start_new_traj = self.episode_horizon == len(trajectory) or done  # Handle terminal steps
        if start_new_traj:
            observation, trajectory = self.handle_completed_trajectory(observation, trajectory)
        else:
            observation = next_observation
        return observation, trajectory

    def handle_completed_trajectory(self,
                                    observation: Observation,
                                    trajectory: Trajectory) -> Tuple[Observation, Trajectory]:
        if len(trajectory) > 0:
            self.episodes += 1
            # Collect returns and sum entropies for monitoring
            returns, entropy = trajectory.trajectory_returns, trajectory.trajectory_entropy
            self.epoch_trajectory_history.append(trajectory)

            # logging
            rewards_msg = f"""Traj #{self.episodes}, Returns {trajectory.trajectory_returns[0][0]} 
            Steps: {len(trajectory)}"""
            print(rewards_msg)
        # self.monitor.record_data(environment.eposide_count, 'reward_' + str(env_id),
        #                          returns, 'scalar')
        # self.monitor.record_data(environment.eposide_count, 'entropy_' + str(env_id),
        #                          entropy, 'scalar')
        # Save completed trajectory and initialise next trajectory

        return self.reset_environment()

    def reset_environment(self) -> Tuple[Observation, Trajectory]:
        return self.env.reset(), Trajectory()

    def learning_step(self, grad_updates_per_learning_step: int):
        ready_to_learn = ((self.memory_buffer.current_size >= self.batch_size and
                           self.memory_buffer.current_size > self.steps_before_learn) and
                          self.agent.is_learning)
        if ready_to_learn:
            for _ in range(grad_updates_per_learning_step):
                batch = self.memory_buffer.sample_batch_transitions(self.batch_size)
                self.learner.learn_from_batch(batch)

    def epoch_start_callback(self):
        self.epoch_start_time = time.time()
        self.epoch_trajectory_history = []

    def epoch_end_callback(self, steps_per_epoch):
        epoch_length = time.time() - self.epoch_start_time
        time_elapsed = f'Time required for {steps_per_epoch} steps: {epoch_length}'
        print(time_elapsed)
