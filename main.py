
import torch
import sys, os
import datetime
sys.path.insert(0, os.path.abspath('..'))
import torch_rl.policies as policies
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_rl.learners as learners
import torch_rl.core as core
from torch_rl.tools import rl_evaluate_policy, rl_evaluate_policy_multiple_times
from torch_rl.policies import DiscreteModelPolicy
from gym.spaces import Discrete
from torch.autograd import Variable

LEARNING_RATE=0.01
STDV=0.01
BATCHES=100
MODEL_DIR='models/'
WORKING_DIR = ''



#===================================================================== CREATION OF THE POLICY =============================================
# Creation of a learning model Q(s): R^N -> R^A

class Net(nn.Module):
    def __init__(self, input_size, action_size, n_friends=0, temperature=1):
        super(Net, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(input_size[0], 6, 3, padding=1) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1   = nn.Linear(16*8*8, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, action_size)
        if n_friends:
            self.fc3a  = nn.Linear(n_friends * (action_size - n_friends), action_size)

        self.temperature = temperature
        self.n_friends = n_friends

    def zero_(self, batch_size=None):
        pass

    def forward(self, x, advice=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        if self.n_friends:
            advice = advice.view(-1, self.num_flat_features(advice))
            advice = F.relu(self.fc3a(advice))
            x += advice

        x = F.softmax(x / self.temperature)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, batch_size, n_friends=0, temperature=1):
        super().__init__()

        h_init = torch.zeros(batch_size, hidden_size)
        self.set_hidden_state(h_init)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size


        self.conv1 = nn.Conv2d(input_size[0], 6, 3, padding=1) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1   = nn.Linear(16*8*8, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        if n_friends:
            self.fc3a  = nn.Linear(n_friends * (action_size - n_friends), 84)

        self.gru = nn.GRUCell(input_size=84, hidden_size=hidden_size)

        self.fc3   = nn.Linear(hidden_size, action_size)

        self.temperature = temperature
        self.n_friends = n_friends

    def forward(self, x, advice=None):

        # embed the state
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # integrate any advice
        if self.n_friends:
            advice = advice.view(-1, self.num_flat_features(advice))
            advice = F.relu(self.fc3a(advice))
            x += advice

        # recurrent part
        self.h = self.gru(x, self.h)

        # output probability distribution over actions
        output = F.softmax(self.fc3(self.h) / self.temperature)
        return output

    def zero_(self, batch_size=None):
        if batch_size is not None:
            self.h.data = torch.zeros(batch_size, self.hidden_size)
            self.batch_size = batch_size
        self.h.data.zero_()
        self.h = Variable(self.h.data)

    def set_hidden_state(self, h):
        self.h = Variable(h)

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Experiment:
    def __init__(self, model, optimizer, envs, epsilon=None, n_train_steps=10000, eval_freq=10, save_freq=100, num_observations=1):

        self.envs = envs
        self.model = model
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
                                torch_model=self.model,
                                optimizer=self.optimizer,
                                epsilon=self.epsilon,
                                num_observations=self.num_observations)
        learning_algorithm.reset()
        def print_train():
            print("Reward = %f (length= %f)"%(learning_algorithm.log.get_last_dynamic_value("avg_total_reward"),learning_algorithm.log.get_last_dynamic_value("avg_length")))

        for step in range(self.n_train_steps):
            self.model.temperature = 10
            learning_algorithm.step(envs=envs,discount_factor=0.95,maximum_episode_length=100)
            print_train()
            
            if step % self.eval_freq == 0:
                self.model.temperature = 1
                policy=learning_algorithm.get_policy()
                r=rl_evaluate_policy_multiple_times(envs[0],policy,100,1.0,10)
                print("Evaluation avg reward = %f "% r)

            if step % self.save_freq == 0:
                model_str = os.path.join(MODEL_DIR, self.unique_id + '.ckpt')
                torch.save(self.model.state_dict(), model_str)
                print("Saved model checkpoint to {}".format(model_str))
        model_str = os.path.join(MODEL_DIR, self.unique_id + '.ckpt')
        torch.save(self.model.state_dict(), model_str)
        print("Saved model checkpoint to {}".format(model_str))
        print("Finished training!")

print("Creating %d environments" % BATCHES)
from environments import MazeEnv1 as Env
#from environments import OneApple as Env

# load friends
friend_model = Net((6, 8, 8), 7)
model_str = os.path.join(MODEL_DIR, 'Mar-5___19-09-56' + '.ckpt')
model_state_dict = torch.load(model_str)
friend_model.load_state_dict(model_state_dict)
friend = DiscreteModelPolicy(Discrete(6), friend_model)
friends = [friend]
#friends = []
num_observations = 1 + len(friends)

envs=[]
for i in range(BATCHES):
    env = Env(friends)
    envs.append(env)

A = envs[0].action_space.n
STATE_SIZE = envs[0].size
print("Number of Actions is: %d" % A)
print("State size is {} x {} x {}".format(*STATE_SIZE))
#model = RecurrentNet(STATE_SIZE, 100, A, BATCHES, n_friends=len(friends))
model = Net(STATE_SIZE, A, n_friends=len(friends))
#model_state_dict = torch.load(MODEL_DIR + '/Mar-3___13-54-59.ckpt')
#model.load_state_dict(model_state_dict)
optimizer= optim.Adam(model.parameters(), lr=LEARNING_RATE)

experiment = Experiment(model, optimizer, envs, n_train_steps=1000, eval_freq=10, save_freq=50, num_observations=num_observations)
experiment.run()
