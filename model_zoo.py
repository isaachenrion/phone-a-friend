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
        #self.input_size = input_size
        #self.output_size = output_size
        #self.num_features = 1
        #for s in self.input_size:
        #    self.num_features *= s

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

class ActionModule(BaseModule):
    def __init__(self, input_size=None, action_size=None, n_friends=None, friend_action_size=None, temperature=1, model_type=None):
        super().__init__(input_size=input_size, output_size=action_size, model_type=model_type)
        self.action_size = action_size
        self.n_friends = n_friends
        self.friend_action_size = friend_action_size
        self.temperature = temperature

        self.update_mask()

    def update_mask(self, actions=None):
        if actions is None:
            actions = [a for a in range(self.action_size)]
        self.mask = torch.zeros(self.action_size, self.action_size)
        for a in actions:
            try: self.mask[a, a] = 1
            except IndexError:
                import ipdb; ipdb.set_trace()
                print("Warning: tried to allow an out-of-bounds action: {}".format(a))
        self.allowed_actions = actions


class FCNet(ActionModule):
    def __init__(self, fc1_dim=200, fc2_dim=200, **kwargs):
        self.bn = kwargs.pop('bn', False)
        super().__init__(model_type='ff', **kwargs)

        fc3_dim = self.action_size
        self.fc1   = nn.Linear(self.num_features, fc1_dim) # an affine operation: y = Wx + b
        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)
        self.fc2   = nn.Linear(fc1_dim, fc2_dim)
        if self.bn:
            self.bn2   = nn.BatchNorm1d(fc2_dim)
        self.fc3   = nn.Linear(fc2_dim + self.n_friends * (self.action_size - self.n_friends), fc3_dim)

        #if self.n_friends:
        #    self.fc3a  = nn.Linear(self.n_friends * (self.action_size - self.n_friends), fc2_dim)

    def forward(self, x, advice=None):
        x = x.view(-1, self.num_flat_features(x))
        if self.bn:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        if self.bn:
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc2(x))

        if self.n_friends:
            advice = advice.view(-1, self.num_flat_features(advice))
            x = torch.cat([advice, x], 1)
        x = self.fc3(x)

        x = torch.mm(F.softmax(x / self.temperature), Variable(self.mask))
        return x


class BaselineNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model_type='ff'
        self.input_size = input_size

        self.num_features = 1
        for s in self.input_size:
            self.num_features *= s

        self.fc1   = nn.Linear(self.num_features, 120) # an affine operation: y = Wx + b
        self.bn1   = nn.BatchNorm1d(120)
        self.fc2   = nn.Linear(120, 84)
        self.bn2   = nn.BatchNorm1d(84)
        self.fc3   = nn.Linear(84, 1)

    def zero_(self, batch_size=None):
        pass

    def forward(self, x, advice=None):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class BaselineNet2(nn.Module):
    def __init__(self, input_size, extra_info_size):
        super().__init__()
        self.model_type='ff'
        self.input_size = input_size
        self.extra_info_size = extra_info_size
        self.spatial_num_features = 1
        for s in self.input_size:
            self.spatial_num_features *= s
        if self.extra_info_size is None:
            self.extra_info_num_features = 0
        else:
            self.extra_info_num_features = self.extra_info_size

        self.fc1   = nn.Linear(self.spatial_num_features + self.extra_info_num_features, 120) # an affine operation: y = Wx + b
        self.bn1   = nn.BatchNorm1d(120)
        self.fc2   = nn.Linear(120, 84)
        self.bn2   = nn.BatchNorm1d(84)
        self.fc3   = nn.Linear(84, 1)

    def zero_(self, batch_size=None):
        pass

    def forward(self, x, advice=None):
        x = x.view(-1, self.num_flat_features(x))
        if advice is not None:
            advice = advice.view(-1, self.num_flat_features(advice))
            x = torch.cat([x, advice], 1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ConvNet(ActionModule):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(input_size[0], 6, 3, padding=1) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1   = nn.Linear(16*8*8, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, action_size)
        if n_friends:
            self.fc3a  = nn.Linear(n_friends * (action_size - n_friends), action_size)

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


class RecurrentConvNet(ActionModule):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(self.input_size[0], 6, 3, padding=1) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1   = nn.Linear(16*8*8, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        if self.n_friends:
            self.fc3a  = nn.Linear(self.n_friends * (self.action_size - self.n_friends), 84)

        self.gru = nn.GRUCell(input_size=84, hidden_size=hidden_size)

        self.fc3   = nn.Linear(hidden_size, self.action_size)


    def forward(self, x, h, advice=None):

        # embed the state
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # integrate any advice
        if self.n_friends:
            advice = advice.view(-1, self.num_flat_features(advice))
            advice = self.fc3a(advice)
            x += advice

        # recurrent part
        h = self.gru(x, h)

        # output probability distribution over actions
        output = F.softmax(self.fc3(h) / self.temperature)
        return output, h


class RecurrentNet(ActionModule):
    def __init__(self, hidden_size, **kwargs):

        self.bn = kwargs.pop('bn', False)
        super().__init__(model_type='recurrent', **kwargs)
        self.hidden_size = hidden_size
        input_size = self.num_features
        if self.n_friends:
            #input_size += self.n_friends * self.friend_action_size
            input_size += self.n_friends
        fc1_dim = 100
        self.fc1   = nn.Linear(input_size, fc1_dim) # an affine operation: y = Wx + b
        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)

        self.gru = nn.GRUCell(input_size=fc1_dim, hidden_size=hidden_size)

        self.fc3   = nn.Linear(hidden_size, self.action_size)


    def forward(self, x, h, advice=None):
        x = x.view(-1, self.num_flat_features(x))
        if self.n_friends:
            advice = advice.view(-1, self.num_flat_features(advice))
            x = torch.cat([x, advice], 1)

        if self.bn:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        h = self.gru(x, h)
        output = torch.mm(F.softmax(self.fc3(h) / self.temperature), Variable(self.mask))
        return output, h


class RecurrentNet2(BaseModule):
    def __init__(self, input_size_dict=None, hidden_size=None, action_size=None, softmax=True, **kwargs):
        self.bn = kwargs.pop('bn', False)
        self.wn = kwargs.pop('wn', False)
        self.softmax = softmax
        super().__init__(model_type='recurrent', output_size=None, input_size=None)
        self.hidden_size = hidden_size
        self.input_size_dict=input_size_dict.copy()
        self.action_size = action_size

        self.spatial_size = self.input_size_dict["spatial"]

        if input_size_dict is None:
            self.extra_info_num_features = 0
        else:
            self.extra_info_num_features = sum([size for key, size in input_size_dict.items() if key != "spatial"])


        self.spatial_num_features = 1

        for s in self.spatial_size:
            self.spatial_num_features *= s

        fc1_dim = 100
        self.use_dict = True
        self.aggregate = False
        if self.aggregate and self.use_dict:
            self.input_size_dict = {"extra_info": self.extra_info_num_features,
                                    "spatial": self.spatial_size}

        if not self.wn:
            if self.use_dict:
                self.fc1 = LinearDict(self.input_size_dict, fc1_dim)
            else:
                self.fc1 = nn.Linear(self.spatial_num_features + self.extra_info_num_features, fc1_dim)
        else:
            if self.use_dict:
                self.fc1 = WeightNormDict(self.input_size_dict, fc1_dim)
            else:
                self.fc1 = WeightNorm(self.spatial_num_features + self.extra_info_num_features, fc1_dim)


        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)
        self.gru = nn.GRUCell(input_size=fc1_dim, hidden_size=hidden_size)
        if not self.wn:
            self.fc3   = nn.Linear(hidden_size, action_size) # an affine operation: y = Wx + b
        else:
            self.fc3   = WeightNorm(hidden_size, action_size)
        self.update_mask()
        #import ipdb; ipdb.set_trace()

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
        spatial_info = state_dict["spatial"].view(-1, self.spatial_num_features)
        extra_info = torch.cat([state.view(-1, self.num_flat_features(state)) for key, state in state_dict.items() if key != "spatial"], 1)

        if self.use_dict:
            if self.aggregate:
                state_dict = {"spatial": spatial_info, "extra_info": extra_info}
            x = self.fc1(state_dict)
        else:
            x = torch.cat([spatial_info, extra_info], 1)
            x = self.fc1(x)
        if self.bn:
            x = F.relu(self.bn1(x))
        else:
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
        self.s1 = Scale(out_size, ini)
        self.b1 = Bias(out_size)

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
            std = ((2 / out_size)**0.5)
            modules.append(Matrix(size, out_size, std=std))
            self.param_keys.append(key)
        self.bias = Bias(out_size)

        self.params = nn.ModuleList(modules)

    def forward(self, state_dict):
        x = torch.cat([state_dict[param_key].view(-1, self.num_flat_features(state_dict[param_key])) for param_key in self.param_keys], 1)
        weight = torch.cat([weight.matrix for weight in self.params], 0)
        x = torch.mm(x, weight)
        x = self.bias(x)
        return x


class AgentNet(ActionModule):
    def __init__(self, hidden_size=None, input_size=None, action_size=None, n_friends=None, bn=None, **kwargs):

        super().__init__(model_type='hrl-rec', input_size=input_size, action_size=action_size + n_friends, n_friends=n_friends, **kwargs)
        base_net = RecurrentNet(hidden_size=hidden_size, friend_action_size = action_size, input_size=input_size, bn=bn, action_size=50, n_friends=n_friends, **kwargs)
        self.base_net = base_net
        self.fca = nn.Linear(50, action_size)
        self.fcd = nn.Linear(50, n_friends + 1)
        #self.action_net = nn.Sequential(self.base_net, self.fca, nn.Softmax())
        #self.decision_net = nn.Sequential(self.base_net, self.fcd, nn.Softmax())
        self.hidden_size = hidden_size

    def forward(self, x, h, advice=None):
        out, h = self.base_net(x, h, advice)
        action_probs = F.softmax(self.fca(out))
        query_probs = F.softmax(self.fcd(out))
        if self.n_friends:
            overall_probs = torch.cat([action_probs * query_probs[:, 0].unsqueeze(1).expand_as(action_probs), query_probs[:, 1:]], 1)
        else:
            overall_probs = action_probs
        return overall_probs, h

class DeciderNet(BaseModule):
    def __init__(self, hidden_size, n_friends=None, friend_action_size=None, **kwargs):
        self.n_friends = n_friends
        self.bn = kwargs.pop('bn', False)
        super().__init__(model_type='recurrent', **kwargs)
        self.hidden_size = hidden_size
        input_size = self.num_features
        if self.n_friends is not None:
            self.friend_action_size = friend_action_size
            input_size += self.n_friends * self.friend_action_size

        fc1_dim = 100
        self.fc1   = nn.Linear(input_size, fc1_dim) # an affine operation: y = Wx + b
        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)

        self.gru = nn.GRUCell(input_size=fc1_dim, hidden_size=hidden_size)

        self.fc3   = nn.Linear(hidden_size, 1)


    def forward(self, x, h, advice=None):
        x = x.view(-1, self.num_flat_features(x))
        if self.n_friends:
            advice = advice.view(-1, self.num_flat_features(advice))
            x = torch.cat([x, advice], 1)

        if self.bn:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        h = self.gru(x, h)
        output = self.fc3(h)
        return output, h


class ActionNet(ActionModule):
    def __init__(self, **kwargs):
        self.bn = kwargs.pop('bn', False)
        super().__init__(**kwargs)
        fc1_dim = 200
        fc2_dim = 200
        #fc2_dim = self.num_features
        fc3_dim = self.action_size
        self.fc1   = nn.Linear(self.num_features, fc1_dim) # an affine operation: y = Wx + b
        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)
        self.fc2   = nn.Linear(fc1_dim, fc2_dim)
        if self.bn:
            self.bn2   = nn.BatchNorm1d(fc2_dim)
        self.fc3   = nn.Linear(fc2_dim, fc3_dim)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        if self.bn:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        if self.bn:
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        action_probs = F.softmax(x / self.temperature)
        return action_probs



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
