import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch
import numpy as np
import collections
from torch.autograd import Variable
#===================================================================== CREATION OF THE POLICY =============================================
# Creation of a learning model Q(s): R^N -> R^A

class BaseModule(nn.Module):
    def __init__(self, input_size=None, output_size=None, model_type=None):
        super().__init__()
        self.model_type = model_type
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, *args):
        pass

    def zero_(self, batch_size=None):
        pass

class RecurrentNet2(BaseModule):
    def __init__(self, input_size_dict=None, hidden_size=None, action_types=None, softmax=True, goal_state=None, **kwargs):
        self.bn = kwargs.pop('bn', False)
        self.wn = kwargs.pop('wn', False)
        self.softmax = softmax
        super().__init__(model_type='recurrent', output_size=None, input_size=None)
        self.hidden_size = hidden_size
        self.input_size_dict=input_size_dict.copy()
        self.action_types = action_types
        self.action_size = len(action_types)
        self.goal_state = goal_state
        fc1_dim = 500
        if not self.wn:
            self.fc1 = LinearDict(self.input_size_dict, fc1_dim)
        else:
            self.fc1 = WeightNormDict(self.input_size_dict, fc1_dim)
        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)
        self.gru = nn.GRUCell(input_size=fc1_dim, hidden_size=hidden_size)
        nn.init.orthogonal(self.gru.weight_hh, gain=1.)
        nn.init.orthogonal(self.gru.weight_ih, gain=1.)
        if not self.wn:
            self.fc3   = nn.Linear(hidden_size, self.action_size) # an affine operation: y = Wx + b
            #self.fc3.weight.data.normal_(0., 1.)
            self.fc3.bias.data.fill_(0.)
        else:
            self.fc3   = WeightNorm(hidden_size, self.action_size)

    def update_mask(self, actions=None):
        if actions is None:
            actions = [a for a in range(self.action_size)]
        self.mask = torch.zeros(self.action_size, self.action_size)
        for a in actions:
            try: self.mask[a, a] = 1
            except IndexError:
                print("Warning: tried to allow an out-of-bounds action: {}".format(a))
        self.allowed_actions = actions

    def forward(self, h, state_dict):
        x = self.fc1(state_dict)
        if self.bn:
            x = self.bn1(x)
        x = F.tanh(x)
        h = self.gru(x, h)
        output = self.fc3(h)
        if self.softmax:
            output = F.softmax(output/10)
        return output, h

class SynthesisNet(BaseModule):
    def __init__(self, input_size_dict=None, hidden_size=None, action_types=None, softmax=True, **kwargs):
        self.bn = kwargs.pop('bn', False)
        self.wn = kwargs.pop('wn', False)
        self.softmax = softmax
        super().__init__(model_type='recurrent', output_size=None, input_size=None)
        self.hidden_size = hidden_size
        self.input_size_dict=input_size_dict.copy()
        self.action_types = action_types
        self.action_size = len(action_types)
        fc1_dim = 20
        if not self.wn:
            self.fc1 = LinearDict(self.input_size_dict, fc1_dim)
        else:
            self.fc1 = WeightNormDict(self.input_size_dict, fc1_dim)
        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)
        self.gru = nn.GRUCell(input_size=fc1_dim, hidden_size=hidden_size)
        if not self.wn:
            self.fc3   = nn.Linear(hidden_size, self.action_size) # an affine operation: y = Wx + b
        else:
            self.fc3   = WeightNorm(hidden_size, self.action_size)

    def forward(self, h, state_dict):
        x = self.fc1(state_dict)
        if self.bn:
            x = self.bn1(x)
        x = F.relu(x)
        h = self.gru(x, h)
        output = self.fc3(h)
        if self.softmax:
            output = F.softmax(output / self.temperature)
        return output, h


class Bias(BaseModule):
    def __init__(self, out_size, initial_bias=0.1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_size).fill_(initial_bias))

    def forward(self, x):
        return x + self.bias.expand_as(x)



class Scale(BaseModule):
    def __init__(self, out_size, initial_scale=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, out_size).fill_(initial_scale))

    def forward(self, x):
        return x * self.scale.expand_as(x)


class Matrix(BaseModule):
    def __init__(self, in_size, out_size, std=1.0):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(in_size, out_size) * std)

    def forward(self, x):
        return torch.mm(x, self.matrix)


class WeightNorm(BaseModule):
    def __init__(self, in_size, out_size, xavier=True):
        super().__init__()
        self.v = Matrix(in_size, out_size)
        if xavier:
            scale = 2 / (in_size)
        else:
            scale = 0.1
        self.s = Scale(out_size, initial_scale=scale)
        self.b = Bias(out_size, initial_bias=0.1)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.v(x)
        x = x / torch.norm(self.v.matrix, 2, 0).expand_as(x)
        x = self.s(x)
        x = self.b(x)
        return x


class WeightNormDict(BaseModule):
    def __init__(self, in_dict, out_size):
        super().__init__()
        modules = list()
        self.param_keys = list()
        for key, size in in_dict.items():
            if type(size) is not int:
                size = np.prod(size).item()
            v = Matrix(size, out_size, std=0.05)
            modules.append(v)
            self.param_keys.append(key)
        self.params = nn.ModuleList(modules)
        self.s1 = Scale(out_size, initial_scale=1/(out_size**0.5))
        self.b1 = Bias(out_size, initial_bias=0.)

    def forward(self, state_dict):
        x = 0.
        v_all = torch.cat([v.matrix for v in self.params], 0)
        for idx, v in enumerate(self.params):
            key = self.param_keys[idx]
            state = state_dict[key]
            temp = state.view(-1, self.num_flat_features(state))
            temp = v(temp)
            x = x + temp
        x = x / torch.norm(v_all, 2, 0).expand_as(x)
        x = self.s1(x)
        x = self.b1(x)
        return x


class LinearDict(BaseModule):
    def __init__(self, in_dict, out_size):
        super().__init__()
        modules = list()
        self.param_keys = list()
        for key, size in in_dict.items():
            if type(size) is not int:
                size = np.prod(size).item()
            std = (2 / out_size**0.5)
            #std = 0.01
            mat = Matrix(size, out_size, std=std)
            nn.init.orthogonal(mat.matrix)
            modules.append(mat)
            self.param_keys.append(key)
        self.bias = Bias(out_size, initial_bias=0.)

        self.params = nn.ModuleList(modules)

    def forward(self, state_dict):
        #for key, s in state_dict.items():
        #    print("Key: {}, size: {}".format(key, s.size()))
        x = torch.cat([state_dict[param_key].view(-1, self.num_flat_features(state_dict[param_key])) for param_key in self.param_keys], 1)
        weight = torch.cat([weight.matrix for weight in self.params], 0)
        x = torch.mm(x, weight)
        x = self.bias(x)
        return x


class Model(nn.Module):
    def __init__(self, batch_size, neural_net, filename=None):
        super().__init__()
        self.neural_net = neural_net
        self.batch_size = batch_size
        self.filename = filename

    def update_mask(self, actions):
        self.neural_net.update_mask(actions)

    def zero_(self, batch_size=None):
        pass

    def forward(self, *args):
        return self.neural_net.forward(*args)

class RecurrentModel(Model):
    def __init__(self, *args):
        super().__init__(*args)
        self.h = Variable(torch.zeros(self.batch_size, self.neural_net.hidden_size))

    def zero_(self, batch_size=None):
        if batch_size is not None:
            self.h.data = torch.zeros(batch_size, self.neural_net.hidden_size)
            self.batch_size = batch_size
        self.h.data.zero_()
        self.h = Variable(self.h.data)

    def forward(self, state_dict):
        output, self.h = self.neural_net(self.h, state_dict)
        return output


class _SynthesisModel(Model):
    def __init__(self, *args):
        super().__init__(*args)
        self.h = dict()
        for k, in_size in self.neural_net.input_size_dict.items():
            self.h[k] = Variable(torch.zeros(self.batch_size, self.neural_net.hidden_size))

    def zero_(self, batch_size=None):
        if batch_size is not None:
            for k, in_size in self.neural_net.input_size_dict.items():
                self.h[k].data = torch.zeros(batch_size, self.neural_net.hidden_size)
            self.batch_size = batch_size
        for k, in_size in self.neural_net.input_size_dict.items():
            self.h[k].data.zero_()
            self.h[k] = Variable(self.h[k].data)

    def forward(self, state_dict):
        output, self.h = self.neural_net(self.h, state_dict)
        return output
