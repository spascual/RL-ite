import random
from typing import Tuple

from gym import Env

from agent import AgentSAC
from config import BasicConfigSAC
from learner import LearnerSAC
from memory_buffer import MemoryBuffer, Trajectory
from monitor import Monitor
from utils import Observation, generate_experiment_signature

random.seed(42)

class TrainingLoopSAC:
    def __init__(self,
                 config: BasicConfigSAC,
                 environment: Env,
                 log_path: str = None,
                 logging: bool = True
                 ):
        """
        TODO: Write docstring
        """
        log_path = generate_experiment_signature(environment) if log_path is None else log_path
        self.config = config
        self.monitor = Monitor(log_path,
                               config,
                               logging=logging)
        self.env = environment

        self.batch_size = config.learner.batch_size
        self.episode_horizon = config.episode_horizon
        self.steps_before_learn = config.steps_before_learn

        self.memory_buffer = MemoryBuffer(max_memory_size=config.memory_size)

        self.agent = AgentSAC(environment, config.policy)

        self.learner = LearnerSAC(
            config=config.learner,
            agent=self.agent,
            enviroment=self.env,
            monitor=self.monitor
        )

    def train(self,
              num_epochs: int = 30,
              steps_per_epoch: int = 1000,
              step_per_learning_step: int = 1,
              grad_updates_per_learning_step: int = 1):

        # assert self.agent.is_learning
        self.total_steps, self.learning_steps, self.episodes = 0, 0, 0

        observation, trajectory = self.reset_environment()

        with self.monitor.summary_writter.as_default():
            for epoch in range(num_epochs):
                self.monitor.epoch_start_callback()
                for step in range(steps_per_epoch):
                    observation, trajectory = self.register_agent_step(observation, trajectory)
                    if step % step_per_learning_step == 0 and self.agent.is_learning:
                        self.learning_step(grad_updates_per_learning_step)
                        self.learning_steps += 1
                self.monitor.epoch_end_callback(steps_per_epoch)

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
        start_new_traj = self.episode_horizon == len(trajectory) or done  # Handle terminal steps
        if start_new_traj:
            observation, trajectory = self.handle_completed_trajectory(trajectory)
        else:
            observation = next_observation
        self.total_steps += 1
        return observation, trajectory

    def handle_completed_trajectory(self,
                                    trajectory: Trajectory) -> Tuple[Observation, Trajectory]:
        if len(trajectory) > 0:
            self.monitor.trajectory_completed_callback(trajectory)
            self.episodes += 1
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
