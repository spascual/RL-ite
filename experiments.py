from gym.envs import box2d

import gym

from config import BasicConfigSAC
from training_loop import TrainingLoopSAC

if __name__ == '__main__':
    config = BasicConfigSAC()
    print(config)
    environment = box2d.BipedalWalker()
    # environment = gym.make("Pendulum-v0")
    sac = TrainingLoopSAC(config, environment, log_path='/tmp/rl-ite/box2d/1')
    sac.train(num_epochs=400, steps_per_epoch=1000)
