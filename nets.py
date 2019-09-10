import tensorflow as tf
import tensorflow_probability as tfp
from gym import Env

from config import BasicPolicyConfigSAC
from utils import Action, State


class GaussianPolicy(tf.Module):
    def __init__(self, environment: Env,  config: BasicPolicyConfigSAC):
        self.is_not_deterministic = False
        layers = [
            tf.keras.layers.Dense(config.hidden_units, activation=config.activation)
            for _ in range(config.hidden_layers)
        ]
        perceptron = tf.keras.Sequential(layers)
        self.model = model
        self.action_dim = model.action_dim


    def sample_action(self, state, act_deterministic=False) -> Action:
        with torch.no_grad():
            # Construct policy input depending on dim of latent space
            state = torch.from_numpy(state).float().view(1, -1)
            if act_deterministic:
                # Restrict to (-1, 1) interval
                torch_action = torch.tanh(self.model(state=state)[0])
            else:
                torch_action, _, log_probs, _, _ = self.model.sample_action(
                    state, reparametrisation=True, include_learner_inputs=True)
        action = torch_action.numpy().reshape(-1)
        chosen_action = ChosenAction(action, prob=1.)
        if not act_deterministic:
            chosen_action.add_metadata('log_probs', log_probs)
        return chosen_action


    def get_mean_action(self, state: State):
        pass

    def __call__(self):


class BoundedGaussianPolicy(GaussianPolicy):
    def __init__(self, environment: Env,  config: BasicPolicyConfigSAC):


class QValueFunction(tf.Module):
    pass

class VValueFunction(tf.Module):
    pass