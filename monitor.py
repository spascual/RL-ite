import time

import tensorflow as tf

from config import BasicConfigSAC
from trajectory import Trajectory


class Monitor:
    def __init__(self,
                 log_path: str,
                 config: BasicConfigSAC,
                 logging: bool = True):
        self.log_path = log_path
        self.config = config
        self.logging = logging
        self.summary_writter = tf.summary.create_file_writer(log_path)
        self.launch_tensorboard_message()

        self.epoch = 0
        self.episodes = 0
        self.learning_step = 0

    def write_to_summary(self, name: str, data: tf.Tensor):
        tf.summary.scalar(name=name, data=data, step=self.learning_step)

    def start_learning_step_callback(self):
        self.learning_step += 1
        if self.logging and self.learning_step % 1000 == 0:
            print(f'Learning step #{self.learning_step}')

    def trajectory_completed_callback(self, trajectory: Trajectory):
        returns, entropy = trajectory.trajectory_returns[0][0], trajectory.trajectory_entropy[0][0]
        tf.summary.scalar('returns', returns, step=self.episodes)
        tf.summary.scalar('entropy', entropy, step=self.episodes)

        if self.logging:
            returns_msg = (f"Traj #{self.episodes}"
                           f" length: {len(trajectory)} / {self.config.episode_horizon}"
                           f" entropy: {entropy: .3f}"
                           f" returns: {returns: .3f}"
                           )
            print(returns_msg)

        self.epoch_trajectory_history.append(trajectory)
        self.episodes += 1

    def epoch_start_callback(self):
        self.epoch_start_time = time.time()
        self.epoch_trajectory_history = []

    def epoch_end_callback(self, steps_per_epoch):
        epoch_length = time.time() - self.epoch_start_time
        time_elapsed = f'Time required for {steps_per_epoch} steps: {epoch_length: .1f}'
        print(time_elapsed)

    def launch_tensorboard_message(self):
        tensorboard_msg = f"Run $ tensorboard --logdir {self.log_path}"
        print(tensorboard_msg)
