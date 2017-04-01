import torch
import sys, os
from constants import MazeConstants as C
from constants import ExperimentConstants as EC
from torch_rl.policies import DiscreteModelPolicy
from model_zoo import RecurrentModel
from torch.autograd import Variable

class Sensor:
    """An object that can provide information about the environment to an agent"""
    def __init__(self, shape, name, wipe_after_call=False):
        self.shape = shape
        self.info = torch.zeros(shape)
        self.name = name
        self.wipe_after_call = wipe_after_call
        self.type=None


    def call(self):
        info = self.info.clone()
        if self.wipe_after_call:
            self.zero_()
        return info

    def zero_(self):
        self.info.zero_()

    def sense(self, state):
        pass

    def reset(self):
        self.zero_()

class RewardSensor(Sensor):
    def __init__(self, reward_key, **kwargs):
        super().__init__(shape=1, name="RewardSensor({})".format(reward_key), **kwargs)
        self.reward_key = reward_key
        self.reward = None
        self.type="reward"

    def attach(self, reward):
        self.reward = reward

    def sense(self, state):
        self.zero_()
        reward_dict = self.reward.reward_dict
        self.info[0] = reward_dict[self.reward_key]

class ActionSensor(Sensor):
    def __init__(self, action_type, **kwargs):
        super().__init__(shape=1, name="ActionSensor({})".format(action_type), **kwargs)
        self.action_type=action_type

    def sense(self, action_type):
        self.zero_()
        if self.action_type == action_type:
            self.info[0] = 1

class PolicySensor(Sensor):
    def __init__(self, model_str, max_action=False, **kwargs):
        out_shape = C.NUM_BASIC_ACTIONS
        self.max_action = max_action
        super().__init__(shape=out_shape, name="PolicySensor({})-{}".format(out_shape, model_str), **kwargs)
        self.type="policy"
        filename = os.path.join(EC.WORKING_DIR, 'experiments', model_str, model_str + '.ckpt')
        net = torch.load(filename)
        policy_model = RecurrentModel(1, net, filename)
        policy_model.eval()
        self.policy = DiscreteModelPolicy(Discrete(C.NUM_BASIC_ACTIONS), policy_model, allowed_actions=range(C.NUM_BASIC_ACTIONS))


    def sense(self, state):
        self.policy.observe({key: Variable(s.unsqueeze(0)) for key, s in state.items()})
        advice, advice_probs = self.policy.sample()
        self.zero_()
        if self.max_action:
            self.info[advice.data[0, 0]] = 1
        else:
            self.info = advice_probs.squeeze(0).data
