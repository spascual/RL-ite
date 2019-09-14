from gym.envs import box2d

from config import BasicConfigSAC
from training_loop import TrainingLoopSAC

if __name__ == '__main__':
    config = BasicConfigSAC()
    print(config)
    environment = box2d.BipedalWalker()
    sac = TrainingLoopSAC(config, environment, log_path=None)
    sac.train(num_epochs=50, steps_per_epoch=100)
