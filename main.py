import torch
import sys, os
import datetime
sys.path.insert(0, os.path.abspath('..'))
import torch_rl.policies as policies
import torch.optim as optim
from torch_rl.learners import TFLog as Log
import torch_rl.learners as learners
import torch_rl.core as core
from torch_rl.tools import rl_evaluate_policy, rl_evaluate_policy_multiple_times
from torch_rl.policies import DiscreteModelPolicy, DiscreteEpsilonGreedyPolicy
from gym.spaces import Discrete
from model_zoo import *
from constants import Constants as C
import argparse
class Experiment:
    def __init__(self, policy, optimizer, envs, epsilon=None, n_train_steps=10000, eval_freq=10, save_freq=100, num_observations=1, debug=False):

        self.envs = envs
        self.policy = policy
        self.optimizer = optimizer
        self.n_train_steps = n_train_steps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.epsilon = epsilon
        self.num_observations = num_observations
        self.debug = debug

        self.do_admin()

    def do_admin(self):
        # make the log directory
        dt = datetime.datetime.now()
        self.unique_id = '{}-{}___{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
        self.model_file = os.path.join(C.MODEL_DIR, self.unique_id + '.ckpt')
        self.experiment_dir = os.path.join(C.WORKING_DIR, 'experiments', self.unique_id)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.log_file = os.path.join(self.experiment_dir, self.unique_id + ".log")
        self.log = Log(self.experiment_dir)
        # write the settings to log
        self.log.add_static_value("batch_size", len(self.envs))
        self.log.add_static_value("n_train_steps", self.n_train_steps)
        self.log.add_static_value("eval_freq", self.eval_freq)
        self.log.add_static_value("save_freq", self.save_freq)
        self.log.add_static_value("model_file", self.model_file)
        self.log.add_static_value('optimizer', self.optimizer)
        self.log.add_static_value('num_observations', self.num_observations)
        self.log.add_static_value("env", self.envs[0].__class__.__name__)
        for k, v in self.policy.info.items():
            self.log.add_static_value('policy_'+k, v)
        for k, v in C.COLLECTED_CONSTANTS.items():
            self.log.add_static_value(k, v)
        self.log.add_static_value("torch_model", self.policy.torch_model)
        self.log.add_static_value("baseline_model", self.policy.baseline_model)
        # get rid of garbage
        self.log.svar.pop("COLLECTED_CONSTANTS")
        self.log.svar.pop("__module__")
        self.log.svar.pop("__qualname__")



    def run(self):
        learning_algorithm=learners.LearnerBatchPolicyGradient(
                                action_space=self.envs[0].action_space,
                                log=self.log,
                                average_reward_window=10,
                                policy=self.policy,
                                optimizer=self.optimizer,
                                epsilon=self.epsilon,
                                num_observations=self.num_observations)
        learning_algorithm.reset()

        def print_train():
            out_str = "Reward = {} (length = {})".format(self.log.get_last_dynamic_value("avg_total_reward"),self.log.get_last_dynamic_value("avg_length"))
            if self.debug:
                out_str += "Query percentage: {}\n".format(learning_algorithm.log.get_last_dynamic_value("avg_queries"))
                if self.policy.baseline_model is not None:
                    out_str += "\nBaseline loss = {}".format(self.log.get_last_dynamic_value("baseline_loss"))
            print(out_str)

        def print_eval():
            out_str = "\nEvaluation after {} training episodes\n Avg reward = {}\n".format(self.log.learner_log.t * len(self.envs), self.log.get_last_dynamic_value("avg_total_reward", eval_mode=True))
            action_breakdown = self.log.get_last_dynamic_value("action_breakdown")
            for idx, value in enumerate(action_breakdown):
                out_str += "Action {} percentage: {}\n".format(idx, value)
            print(out_str)

        for step in range(self.n_train_steps):
            self.policy.torch_model.temperature = C.TRAIN_TEMPERATURE
            self.policy.torch_model.train()
            learning_algorithm.step(envs=envs,discount_factor=0.95,maximum_episode_length=C.EPISODE_LENGTH)
            print_train()

            if step > 0:
                self.log.log()
            if step % self.eval_freq == 0:
                self.policy.torch_model.eval()
                policy=learning_algorithm.get_greedy_policy()
                _=rl_evaluate_policy_multiple_times(envs[0],self.log,policy,C.EPISODE_LENGTH,1.0,10)
                print_eval()
                if step > 0:
                    self.log.log(eval_mode=True)

            if step + 1 % self.save_freq == 0:
                torch.save(self.log.learner_log, self.log_file)
                torch.save(self.policy.torch_model.neural_net, self.model_file)
                print("Saved model checkpoint to {}".format(self.model_file))
            if step == 0:
                self.log.create_summaries()



        torch.save(self.policy.torch_model.neural_net, self.model_file)
        torch.save(self.log.learner_log, self.log_file)
        print("Saved model checkpoint to {}".format(self.model_file))
        print("Finished training!")


parser = argparse.ArgumentParser(description='Phone a friend')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--env', '-e', type=int, default=0,
                    help='index of environment (default = 0)')
parser.add_argument('--model_dir', type=str, default=C.MODEL_DIR)
parser.add_argument('--n_train_steps', '-n', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

'''LOAD FRIENDS'''
model_strs = ['Mar-8___16-29-54.ckpt']
model_strs = []
friends = []
for model_str in model_strs:
    filename = os.path.join(args.model_dir, model_str)
    friend_model = Model(1, torch.load(filename))
    friend = DiscreteModelPolicy(Discrete(C.NUM_BASIC_ACTIONS), friend_model, disallowed_actions=[act for act in range(C.NUM_BASIC_ACTIONS, C.NUM_BASIC_ACTIONS+len(model_strs))])
    friends.append(friend)

num_observations = 1 + len(friends)

''' LOAD THE ENVIRONMENTS '''
print("Creating %d environments" % args.batch_size)
from environments import *

env_list = [
ApplesEverywhere,
Basic,
NoWalls,
RandomFruit,
OneApple,
EasyExit,
RandomApple,
RandomOrange,
RandomPear
]
Env = env_list[args.env]
print("Environment: {}".format(Env.__name__))

envs=[Env(friends=friends) for _ in range(args.batch_size)]
A = envs[0].action_space.n
state_size = envs[0].size
print("Number of Actions is: %d" % A)
print("State size is {} x {} x {}".format(*state_size))

''' BUILD MODEL '''
disallowed_actions=[]

recurrent = 0
if recurrent:
    net = RecurrentNet(100, state_size, A, n_friends=len(friends), disallowed_actions=disallowed_actions)
    torch_model = RecurrentModel(args.batch_size, net)
else:
    net = FCNet(state_size, A, n_friends=len(friends), disallowed_actions=disallowed_actions)
    torch_model = Model(args.batch_size, net)
    baseline_net = BaselineNet(state_size, 1)
    baseline_model = Model(args.batch_size, baseline_net)
    #baseline_model = None

policy = DiscreteModelPolicy(envs[0].action_space, torch_model, stochastic=True, disallowed_actions=disallowed_actions, baseline_model=baseline_model)
policy = DiscreteEpsilonGreedyPolicy(envs[0].action_space, epsilon=0.1, policy=policy)
optimizer= optim.Adam(policy.parameters(), lr=args.lr)

''' RUN EXPERIMENT '''
experiment = Experiment(policy, optimizer, envs, n_train_steps=1000, eval_freq=10, save_freq=50, num_observations=num_observations, debug=args.debug)
experiment.run()
