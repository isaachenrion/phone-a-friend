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
from experiment import Experiment
import argparse

parser = argparse.ArgumentParser(description='Phone a friend')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--env', '-m', type=int, default=0,
                    help='index of environment (default = 0)')
parser.add_argument('--n_train_steps', '-n', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--debug','-d', action='store_true')
parser.add_argument('--recurrent','-r', action='store_true')
parser.add_argument('--entropy_term', '-z', type=float, default=1)
parser.add_argument('--clip', '-c', action='store_true')
parser.add_argument('--use_friends', '-f', action='store_true')
parser.add_argument('--epsilon', '-e', type=float, default=0.0)
parser.add_argument('--load', '-l', type=str, default=None)
parser.add_argument('--no_bn', action='store_true')
args = parser.parse_args()
args.bn = not args.no_bn

'''LOAD FRIENDS'''
model_strs = []
if args.use_friends:
    model_strs.append('Mar-14___19-34-35-RandomPear-recurrent')
    model_strs.append('Mar-15___10-50-29-RandomOrange-recurrent')
    model_strs.append('Mar-15___10-51-59-RandomApple-recurrent')
for s in model_strs:
    print("Loaded friend: {}".format(s))
friends = []
for model_str in model_strs:
    filename = os.path.join(C.WORKING_DIR, 'experiments', model_str, model_str + '.ckpt')
    net = torch.load(filename)
    if net.model_type == 'ff':
        friend_model = Model(1, net, filename)
    elif net.model_type == 'recurrent':
        friend_model = RecurrentModel(1, net, filename)
    friend_model.eval()
    friend = DiscreteModelPolicy(Discrete(C.NUM_BASIC_ACTIONS), friend_model, allowed_actions=range(C.NUM_BASIC_ACTIONS))
    friends.append(friend)

num_observations = 1 + (len(friends) != 0)

''' LOAD ENVIRONMENTS '''
print("Creating %d environments" % args.batch_size)
from environments.env_list import ENVS
Env = ENVS[args.env]
print("Environment: {}".format(Env.__name__))

#allowed_actions=[5, 6, 7]
allowed_actions = range(C.NUM_BASIC_ACTIONS + len(friends))
envs=[Env(friends=friends) for _ in range(args.batch_size)]
A = envs[0].action_space.n
state_size = envs[0].size
print("Number of Actions is: {}".format(A))
print("State size is {} x {} x {}".format(*state_size))

''' BUILD MODEL '''
baseline_net = BaselineNet(state_size)
baseline_model = Model(args.batch_size, baseline_net)

if args.recurrent:
    action_net = RecurrentNet(50, state_size, A, n_friends=len(friends), bn=args.bn)
    action_model = RecurrentModel(args.batch_size, action_net)
else:
    action_net = FCNet(state_size, A, n_friends=len(friends), bn=args.bn)
    action_model = Model(args.batch_size, action_net)

if args.load is not None:
    print("Loading model: {}".format(args.load))
    filename = os.path.join(C.WORKING_DIR, 'experiments', args.load, args.load + '.ckpt')
    if args.recurrent:
        action_model = RecurrentModel(1, torch.load(filename), filename)
    else:
        action_model = Model(1, torch.load(filename), filename)

print("Action model: {}".format(action_net))
print("Baseline model: {}".format(baseline_net))

policy = DiscreteModelPolicy(envs[0].action_space, action_model, stochastic=True, allowed_actions=allowed_actions, baseline_model=baseline_model)
#policy = DiscreteEpsilonGreedyPolicy(envs[0].action_space, epsilon=args.epsilon, policy=policy)
optimizer= optim.Adam(policy.parameters(), lr=args.lr)
#optimizer= optim.RMSprop(policy.parameters(), lr=args.lr)
#optimizer = optim.SGD(policy.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

''' RUN EXPERIMENT '''
print("Running experiment...")
experiment = Experiment(policy, optimizer, envs, n_train_steps=1000, eval_freq=10, save_freq=50, num_observations=num_observations, args=args)
try:
    experiment.run()
except (KeyboardInterrupt, SystemExit):
    # quit
    save = input("Save model? y/n\n")
    while save not in ['y', 'n']:
        save = input("Save model? y/n\n")
    if save == 'y':
        experiment.save_everything()
    elif save == 'n':
        experiment.kill()
    else: raise ValueError("Enter y or n")
    sys.exit()
