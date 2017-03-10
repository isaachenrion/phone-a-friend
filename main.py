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
    def __init__(self, policy, optimizer, envs, epsilon=None, n_train_steps=10000, eval_freq=10, save_freq=100, num_observations=1, args=None):

        self.envs = envs
        self.policy = policy
        self.optimizer = optimizer
        self.n_train_steps = n_train_steps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.epsilon = epsilon
        self.num_observations = num_observations
        self.args = args
        #import ipdb; ipdb.set_trace()
        self.do_admin()

    def do_admin(self):
        # make the log directory
        dt = datetime.datetime.now()
        self.unique_id = '{}-{}___{:02d}-{:02d}-{:02d}-{}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second, self.envs[0].__class__.__name__)
        self.model_file = os.path.join(C.MODEL_DIR, self.unique_id + '.ckpt')
        self.experiment_dir = os.path.join(C.WORKING_DIR, 'experiments', self.unique_id)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.log_file = os.path.join(self.experiment_dir, self.unique_id + ".log")
        self.log = Log(self.experiment_dir)
        torch.save(self.log.learner_log, self.log_file)

        # write the settings to log
        def write_and_log(k, v, f, log):
            f.write("{}: {}\n".format(k, v))
            log.add_static_value(k, v)
        log = self.log
        with open(os.path.join(self.experiment_dir,'settings.txt'), 'w') as f:
            f.write("####### EXPERIMENT PARAMETERS #######\n\n")
            write_and_log("eval_freq", self.eval_freq, f, log)
            write_and_log("save_freq", self.save_freq, f, log)
            write_and_log("model_file", self.model_file, f, log)
            write_and_log('optimizer', self.optimizer, f, log)
            write_and_log('num_observations', self.num_observations, f, log)
            write_and_log("env_name", self.envs[0].__class__.__name__, f, log)
            for idx, friend in enumerate(self.envs[0].friends):
                write_and_log("Friend{}".format(idx + 1), friend.action_model.filename, f, log)
            for k, v in vars(self.args).items():
                write_and_log(k, v, f, log)

            f.write("\n####### POLICY #######\n\n")
            for k, v in self.policy.info.items():
                write_and_log('policy_'+k, v, f, log)
            write_and_log("action_model", self.policy.action_model, f, log)
            write_and_log("baseline_model", self.policy.baseline_model, f, log)

            f.write("\n####### CONSTANTS #######\n\n")
            # get rid of garbage
            garbage = ["COLLECTED_CONSTANTS", "__module__", "__qualname__"]
            for k, v in C.COLLECTED_CONSTANTS.items():
                if k not in garbage:
                    write_and_log(k, v, f, log)




    def print_train(self):
        out_str = "Reward = {} (length = {})".format(self.log.get_last_dynamic_value("avg_total_reward"),self.log.get_last_dynamic_value("avg_length"))
        if self.args.debug:
            out_str += "\nQuery percentage: {}".format(self.log.get_last_dynamic_value("avg_queries"))
            if self.policy.baseline_model is not None:
                out_str += "\nBaseline loss = {}".format(self.log.get_last_dynamic_value("baseline_loss"))
        print(out_str)

    def print_eval(self):
        out_str = "\nEvaluation after {} training episodes\nAvg reward = {}".format(self.log.learner_log.t * len(self.envs), self.log.get_last_dynamic_value("avg_total_reward", eval_mode=True))
        action_breakdown = self.log.get_last_dynamic_value("action_breakdown")
        for idx, value in enumerate(action_breakdown):
            out_str += "\nAction {} percentage: {}".format(idx, value)
        print(out_str)

    def save_everything(self):
        torch.save(self.log.learner_log, self.log_file)
        torch.save(self.policy.action_model.neural_net, self.model_file)
        print("Saved model checkpoint to {}".format(self.model_file))

    def train(self, step, learning_algorithm):
        self.policy.action_model.temperature = C.TRAIN_TEMPERATURE
        self.policy.action_model.train()
        learning_algorithm.step(envs=envs,discount_factor=0.95,maximum_episode_length=C.EPISODE_LENGTH)
        self.print_train()
        if step > 0:
            self.log.log()

    def evaluate(self, step, learning_algorithm):
        self.policy.action_model.eval()
        policy=learning_algorithm.get_greedy_policy()
        _=rl_evaluate_policy_multiple_times(self.envs[0],self.log,policy,C.EPISODE_LENGTH,1.0,10)
        self.print_eval()
        if step > 0:
            self.log.log(eval_mode=True)

    def run(self):
        learning_algorithm=learners.LearnerBatchPolicyGradient(
                                action_space=self.envs[0].action_space,
                                log=self.log,
                                average_reward_window=10,
                                policy=self.policy,
                                optimizer=self.optimizer,
                                epsilon=self.epsilon,
                                num_observations=self.num_observations,
                                args=self.args)
        learning_algorithm.reset()

        for step in range(self.n_train_steps):
            self.train(step, learning_algorithm)
            if step % self.eval_freq == 0:
                self.evaluate(step, learning_algorithm)
            if step % self.save_freq == 0:
                self.save_everything()
            if step == 0:
                self.log.create_summaries()

        self.save_everything()
        print("Finished training!")


parser = argparse.ArgumentParser(description='Phone a friend')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--env', '-m', type=int, default=0,
                    help='index of environment (default = 0)')
parser.add_argument('--model_dir', type=str, default=C.MODEL_DIR)
parser.add_argument('--n_train_steps', '-n', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--debug','-d', action='store_true')
parser.add_argument('--recurrent','-r', action='store_true')
parser.add_argument('--entropy_term', '-z', type=float, default=0.01)
parser.add_argument('--clip', '-c', action='store_true')
parser.add_argument('--use_friends', '-f', action='store_true')
parser.add_argument('--epsilon', '-e', type=float, default=0.1)
args = parser.parse_args()

'''LOAD FRIENDS'''
model_strs = []
if args.use_friends:
    #model_strs.append('Mar-10___11-02-01-NoWalls.ckpt')
    model_strs.append('Mar-10___15-53-37-RandomApple.ckpt')
    model_strs.append('Mar-10___15-41-14-RandomOrange.ckpt')
    model_strs.append('Mar-10___15-50-53-RandomPear.ckpt')
for s in model_strs:
    print("Loaded friend: {}".format(s))
friends = []
for model_str in model_strs:
    filename = os.path.join(args.model_dir, model_str)
    friend_model = Model(1, torch.load(filename), filename)
    friend = DiscreteModelPolicy(Discrete(C.NUM_BASIC_ACTIONS), friend_model, disallowed_actions=[act for act in range(C.NUM_BASIC_ACTIONS, C.NUM_BASIC_ACTIONS+len(model_strs))])
    friends.append(friend)

num_observations = 1 + (len(friends) != 0)

''' LOAD ENVIRONMENTS '''
print("Creating %d environments" % args.batch_size)
from environments.env_list import ENVS
Env = ENVS[args.env]
print("Environment: {}".format(Env.__name__))

disallowed_actions=[5, 6, 7]
envs=[Env(friends=friends, disallowed_actions=disallowed_actions) for _ in range(args.batch_size)]
A = envs[0].action_space.n
state_size = envs[0].size
print("Number of Actions is: {}".format(A)
print("State size is {} x {} x {}".format(*state_size))

''' BUILD MODEL '''
baseline_net = BaselineNet(state_size, 1)
baseline_model = Model(args.batch_size, baseline_net)

if args.recurrent:
    action_net = RecurrentNet(100, state_size, A, n_friends=len(friends), disallowed_actions=disallowed_actions)
    action_model = RecurrentModel(args.batch_size, action_net)
else:
    action_net = FCNet(state_size, A, n_friends=len(friends), disallowed_actions=disallowed_actions)
    action_model = Model(args.batch_size, action_net)

print("Action model: {}".format(action_net))
print("Baseline model: {}".format(baseline_net))

policy = DiscreteModelPolicy(envs[0].action_space, action_model, stochastic=True, disallowed_actions=disallowed_actions, baseline_model=baseline_model)
policy = DiscreteEpsilonGreedyPolicy(envs[0].action_space, epsilon=args.epsilon, policy=policy)
optimizer= optim.Adam(policy.parameters(), lr=args.lr)

''' RUN EXPERIMENT '''
print("Running experiment...")
experiment = Experiment(policy, optimizer, envs, n_train_steps=1000, eval_freq=10, save_freq=50, num_observations=num_observations, args=args)
experiment.run()
