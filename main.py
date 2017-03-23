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
from constants import ExperimentConstants as C
from constants import MazeConstants as MC
from experiment import Experiment
from environments.maze import PolicySensor, RewardSensor
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
parser.add_argument('--ff','-f', action='store_true')
parser.add_argument('--entropy_term', '-z', type=float, default=1)
parser.add_argument('--clip', '-c', type=float, default=1.0)
parser.add_argument('--use_policy_sensors', '-p', action='store_true')
parser.add_argument('--use_reward_sensors', '-r', action='store_true')
parser.add_argument('--epsilon', '-e', type=float, default=0.0)
parser.add_argument('--load', '-l', type=str, default=None)
parser.add_argument('--no_bn', action='store_true')
parser.add_argument('--wn', action='store_true')
args = parser.parse_args()
args.bn = not args.no_bn
args.recurrent = not args.ff

def main():
    '''LOAD FRIENDS'''
    active_sensors = [None] * args.batch_size
    for idx in range(args.batch_size):
        policy_sensors = []
        if args.use_policy_sensors:
            model_strs = []
            #model_strs.append('Mar-22___16-15-34-RandomPear-recurrent') # BAD PEAR
            model_strs.append('Mar-23___14-13-58-RandomPear-recurrent') # GOOD PEAR
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

        num_active_sensors = len(policy_sensors) + len(reward_sensors)
        if idx == 0:
            print("Total number of active sensors: {}".format(num_active_sensors))
        if num_active_sensors > 0:
            active_sensors[idx] = policy_sensors + reward_sensors
        else:
            active_sensors[idx] = None

    ''' LOAD ENVIRONMENTS '''
    print("Creating %d environments" % args.batch_size)
    from environments.env_list import ENVS
    Env = ENVS[args.env]
    print("Environment: {}".format(Env.__name__))


    #import ipdb; ipdb.set_trace()
    #allowed_actions=[5, 6, 7]
    allowed_actions = range(MC.NUM_BASIC_ACTIONS + num_active_sensors)
    envs=[Env(active_sensors=active_sensors[idx]) for idx in range(args.batch_size)]

    A = envs[0].action_space.n
    try:
        E = sum([sensor.shape for sensor in envs[0].world.agent.sensors])

    except TypeError:
        E = None

    print("Total sensor size: {}".format(E))
    state_size_dict = envs[0].world.state_size_dict

    print("Number of Actions is: {}".format(A))
    print("Spatial state size is {} x {} x {}".format(*state_size_dict["spatial"]))

    ''' BUILD MODEL '''

    num_observations = 1 + (E is not None)

    if MC.EXPERIMENTAL:
        baseline_net = RecurrentNet2(input_size_dict=state_size_dict, hidden_size=20, action_size=1, softmax=False, bn=args.bn, wn=args.wn)
        baseline_model = RecurrentModel(args.batch_size, baseline_net)
    else:
        baseline_net = BaselineNet(state_size)
        baseline_model = Model(args.batch_size, baseline_net)

    if args.recurrent:
        action_net = RecurrentNet2(hidden_size=50, input_size_dict=state_size_dict, action_size=A, bn=args.bn, wn=args.wn)

        action_model = RecurrentModel(args.batch_size, action_net)
    else:
        raise ValueError("MUST CHANGE FCNET IMPLEMENTATION")
        action_net = FCNet(input_size=state_size, action_size=A, n_friends=len(friends), bn=args.bn)
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
    experiment._run()

main()
