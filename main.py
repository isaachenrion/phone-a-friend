import torch
import sys, os
import datetime
sys.path.insert(0, os.path.abspath('..'))
import torch_rl.policies as policies
import torch.optim as optim
import torch_rl.learners as learners
import torch_rl.core as core
from torch_rl.tools import rl_evaluate_policy, rl_evaluate_policy_multiple_times
from torch_rl.policies import DiscreteModelPolicy, DiscreteEpsilonGreedyPolicy
from gym.spaces import Discrete
from model_zoo import *
from constants import *

class Experiment:
    def __init__(self, policy, optimizer, envs, epsilon=None, n_train_steps=10000, eval_freq=10, save_freq=100, num_observations=1):

        self.envs = envs
        self.policy = policy
        self.optimizer = optimizer
        self.n_train_steps = n_train_steps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.epsilon = epsilon
        self.num_observations = num_observations
        self.do_admin()

    def do_admin(self):
        # make the log directory
        dt = datetime.datetime.now()
        self.unique_id = '{}-{}___{:02d}-{:02d}-{:02d}'.format(dt.strftime("%b"), dt.day, dt.hour, dt.minute, dt.second)
        self.experiment_dir = os.path.join(WORKING_DIR, 'experiments', self.unique_id)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.experiment_log = os.path.join(self.experiment_dir, "log.txt")

    def run(self):
        learning_algorithm=learners.LearnerBatchPolicyGradient(
                                action_space=self.envs[0].action_space,
                                average_reward_window=10,
                                policy=self.policy,
                                optimizer=self.optimizer,
                                epsilon=self.epsilon,
                                num_observations=self.num_observations)
        learning_algorithm.reset()

        def print_train():
            out_str = "Reward = {} (length = {}".format(learning_algorithm.log.get_last_dynamic_value("avg_total_reward"),learning_algorithm.log.get_last_dynamic_value("avg_length"))
            #out_str += "Query percentage: {}\n".format(learning_algorithm.log.get_last_dynamic_value("avg_queries"))
            print(out_str)

        def print_eval():
            out_str = "\nEvaluation avg reward = {}\n".format(r)
            action_breakdown = learning_algorithm.log.get_last_dynamic_value("action_breakdown")
            for idx, value in enumerate(action_breakdown):
                out_str += "Action {} percentage: {}\n".format(idx, value)
            print(out_str)

        for step in range(self.n_train_steps):
            self.policy.torch_model.temperature = 10
            self.policy.torch_model.train()
            learning_algorithm.step(envs=envs,discount_factor=0.95,maximum_episode_length=EPISODE_LENGTH)
            print_train()

            if step % self.eval_freq == 0:
                self.policy.torch_model.eval()
                policy=learning_algorithm.get_greedy_policy()

                r=rl_evaluate_policy_multiple_times(envs[0],policy,EPISODE_LENGTH,1.0,10)
                print_eval()

            if step % self.save_freq == 0:
                model_str = os.path.join(MODEL_DIR, self.unique_id + '.ckpt')
                torch.save(self.policy.torch_model.neural_net, model_str)
                print("Saved model checkpoint to {}".format(model_str))
        model_str = os.path.join(MODEL_DIR, self.unique_id + '.ckpt')
        torch.save(self.policy.torch_model.neural_net, model_str)
        print("Saved model checkpoint to {}".format(model_str))
        print("Finished training!")


'''LOAD FRIENDS'''
model_strs = []
#model_strs.append('Mar-7___10-37-21.ckpt')
friends = []
for model_str in model_strs:
    filename = os.path.join(MODEL_DIR, model_str)
    friend_model = RecurrentModel(1, torch.load(filename))
    friend = DiscreteModelPolicy(Discrete(NUM_BASIC_ACTIONS), friend_model, disallowed_actions=[act for act in range(NUM_BASIC_ACTIONS, NUM_BASIC_ACTIONS+len(model_strs))])
    friends.append(friend)

num_observations = 1 + len(friends)

''' LOAD THE ENVIRONMENTS '''
print("Creating %d environments" % BATCHES)
from environments import *

Env = ApplesEverywhere
#Env = Basic
#Env = NoWalls
#Env = RandomFruit
#Env = OneApple
#Env = EasyExit
envs=[]
for i in range(BATCHES):
    env = Env(friends=friends)
    envs.append(env)

A = envs[0].action_space.n
STATE_SIZE = envs[0].size
print("Number of Actions is: %d" % A)
print("State size is {} x {} x {}".format(*STATE_SIZE))

''' BUILD MODEL '''
disallowed_actions=[]

recurrent = 0
if recurrent:
    net = RecurrentNet(STATE_SIZE, 100, A, n_friends=len(friends), disallowed_actions=disallowed_actions)
    torch_model = RecurrentModel(BATCHES, net)
else:
    net = FCNet(STATE_SIZE, A, n_friends=len(friends), disallowed_actions=disallowed_actions)
    torch_model = Model(BATCHES, net)

optimizer= optim.Adam(torch_model.parameters(), lr=LEARNING_RATE)
policy = DiscreteModelPolicy(envs[0].action_space, torch_model, stochastic=True, disallowed_actions=disallowed_actions)
policy = DiscreteEpsilonGreedyPolicy(envs[0].action_space, epsilon=0.1, policy=policy)

''' RUN EXPERIMENT '''
experiment = Experiment(policy, optimizer, envs, n_train_steps=1000, eval_freq=10, save_freq=50, num_observations=num_observations)
experiment.run()
