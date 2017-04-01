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
from torch_rl.policies import DiscreteModelPolicy, DiscreteEpsilonGreedyPolicy, CollectionOfPolicies
from gym.spaces import Discrete
from model_zoo import *
from constants import ExperimentConstants as C
from constants import MazeConstants as MC
from experiment import Experiment
from environments.maze.sensor import PolicySensor, RewardSensor
from environments.maze.agent import Subordinate
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Phone a friend')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--env', '-m', type=int, default=0,
                    help='index of environment (default = 0)')
parser.add_argument('--n_train_steps', '-n', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--debug','-d', action='store_true')
parser.add_argument('--ff','-f', action='store_true')
parser.add_argument('--entropy_term', '-z', type=float, default=1)
parser.add_argument('--clip', '-c', type=float, default=1.0)
parser.add_argument('--use_policy_sensors', '-p', action='store_true')
parser.add_argument('--use_reward_sensors', '-r', action='store_true')
parser.add_argument('--use_subordinates', '-s', action='store_true')
parser.add_argument('--epsilon', '-e', type=float, default=0.0)
parser.add_argument('--load', '-l', type=str, default=None)
parser.add_argument('--no_bn', action='store_true')
parser.add_argument('--wn', action='store_true')
parser.add_argument('--mp', action='store_true')

args = parser.parse_args()
args.bn = not args.no_bn
args.recurrent = not args.ff
args.multiprocessing = args.mp

def main():
    '''LOAD FRIENDS'''
    active_sensors = [None] * args.batch_size
    subordinates_batch = [None] * args.batch_size
    for idx in range(args.batch_size):
        subordinates = {}
        if args.use_subordinates:
            model_strs = []
            model_strs.append('Mar-27___12-12-19-RandomPear-recurrent') # BAD PEAR
            #model_strs.append('Mar-24___16-14-48-RandomPear-recurrent') # GOOD PEAR
            for s in model_strs:
                subordinates[s] = Subordinate(s)
                if idx == 0:
                    print("Loaded subordinate: {}".format(s))
        num_subordinates = len(subordinates)
        if num_subordinates > 0:
            subordinates_batch[idx] = subordinates

        policy_sensors = []
        if args.use_policy_sensors:
            model_strs = []
            #model_strs.append('Mar-23___14-33-50-RandomPear-recurrent') # BAD PEAR
            model_strs.append('Mar-24___16-14-48-RandomPear-recurrent') # GOOD PEAR
            for s in model_strs:
                policy_sensors.append(PolicySensor(s))
                if idx == 0:
                    print("Loaded policy sensor: {}".format(s))

        reward_sensors = []
        if args.use_reward_sensors:
            reward_strs = ['pear', 'orange', 'apple']
            for s in reward_strs:
                reward_sensors.append(RewardSensor(s))
                if idx == 0:
                    print("Loaded reward sensor: {}".format(s))

        if idx == 0:
            num_active_sensors = len(policy_sensors) + len(reward_sensors)
            print("Total number of active sensors: {}".format(num_active_sensors))
        if num_active_sensors > 0:
            active_sensors[idx] = policy_sensors + reward_sensors

    ''' LOAD ENVIRONMENTS '''
    print("Creating %d environments" % args.batch_size)
    from environments.maze.env_list import ENVS
    Env = ENVS[args.env]
    print("Environment: {}".format(Env.__name__))

    #allowed_actions = range(MC.NUM_BASIC_ACTIONS + num_active_sensors + num_subordinates)
    policy_dict = {}
    model_strs = []
    #model_strs.append('Mar-23___14-33-50-RandomPear-recurrent') # BAD PEAR
    #model_strs.append('Mar-27___14-26-20-RandomPear-recurrent') # GOOD PEAR

    for s in model_strs:
        print("Loading model: {}".format(s))
        filename = os.path.join(C.WORKING_DIR, 'experiments', s, s + '.ckpt')
        model = RecurrentModel(1, torch.load(filename), filename)
        policy_dict[s] = DiscreteModelPolicy(model, stochastic=True, baseline_model=None, batch_size=args.batch_size)

    envs=[Env(active_sensors=active_sensors[idx], subordinates=model_strs, seed=idx) for idx in range(args.batch_size)]

    A = envs[0].action_space.n
    action_types = envs[0].world.agent.action_types
    goal_state = envs[0].reward.goal_state
    try:
        E = sum([sensor.shape for sensor in envs[0].world.agent.sensors])

    except TypeError:
        E = None

    print("Total sensor size: {}".format(E))
    state_size_dict = envs[0].world.state_size_dict

    print("Number of Actions is: {}".format(A))

    ''' BUILD MODEL '''

    if args.multiprocessing:
        bs = 1
    else: bs = args.batch_size

    baseline_net = RecurrentNet2(input_size_dict=state_size_dict, hidden_size=20, action_types=["value"], softmax=False, bn=args.bn, wn=args.wn)
    baseline_model = RecurrentModel(bs, baseline_net)
    #baseline_model= None

    _action_net = RecurrentNet2(hidden_size=50, input_size_dict=state_size_dict, action_types=action_types, goal_state=goal_state, bn=args.bn, wn=args.wn)
    _action_model = RecurrentModel(bs, _action_net)


    if args.load is not None:
        print("Loading model: {}".format(args.load))
        filename = os.path.join(C.WORKING_DIR, 'experiments', args.load, args.load + '.ckpt')
        _action_net = torch.load(filename)

        _action_model = RecurrentModel(1, _action_net, filename)

    if MC.EXPERIMENTAL and False:
        policy = DiscreteModelPolicy(_action_model, stochastic=True, baseline_model=baseline_model)
        policy_dict["main_policy"] = policy
        policy = CollectionOfPolicies(policy_dict, envs)
    else:
        action_model = _action_model
        policy = DiscreteModelPolicy(action_model, stochastic=True, baseline_model=baseline_model)


    print("Action model: {}".format(_action_net))
    print("Baseline model: {}".format(baseline_net))

    optimizer= optim.Adam(policy.parameters(), lr=args.lr)
    #optimizer= optim.RMSprop(policy.parameters(), lr=args.lr)

    ''' RUN EXPERIMENT '''
    print("Running experiment...")
    print("There are {} CPUs on this machine".format(multiprocessing.cpu_count()))
    experiment = Experiment(policy, optimizer, envs, n_train_steps=1000, eval_freq=10, save_freq=50, args=args)
    experiment._run()

if __name__ == "__main__":
    main()
