import torch.nn as nn
import torch.nn.functional as F
import torch
from constants import Constants as C
from torch.autograd import Variable
#===================================================================== CREATION OF THE POLICY =============================================
# Creation of a learning model Q(s): R^N -> R^A
class Module(nn.Module):
    def __init__(self, input_size, action_size, n_friends=0, temperature=1, disallowed_actions=None):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.n_friends = n_friends
        self.temperature = temperature
        self.disallowed_actions = disallowed_actions if disallowed_actions is not None else []
        self.mask = torch.eye(action_size, action_size)

        self.num_features = 1
        for s in self.input_size:
            self.num_features *= s
        for a in self.disallowed_actions:
            try: self.mask[a, a] = 0
            except IndexError:
                print("Warning: tried to disallow an out-of-bounds action: {}".format(a))


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

class FCNet(Module):
    def __init__(self, *args, **kwargs):
        self.bn = kwargs.pop('bn', False)
        super().__init__(*args, **kwargs)
        fc1_dim = 200
        fc2_dim = 200
        fc3_dim = self.action_size
        self.fc1   = nn.Linear(self.num_features, fc1_dim) # an affine operation: y = Wx + b
        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)
        self.fc2   = nn.Linear(fc1_dim, fc2_dim)
        if self.bn:
            self.bn2   = nn.BatchNorm1d(fc2_dim)
        self.fc3   = nn.Linear(fc2_dim + self.n_friends * (self.action_size - self.n_friends), fc3_dim)

        if self.n_friends:
            self.fc3a  = nn.Linear(self.n_friends * (self.action_size - self.n_friends), fc3_dim)

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
            #if advice.data.sum() > 0:
            #    import ipdb; ipdb.set_trace()
            #import ipdb; ipdb.set_trace()
            #advice = self.fc3a(advice)
        x = self.fc3(x)

        x = torch.mm(F.softmax(x / self.temperature), Variable(self.mask))
        return x


class BaselineNet(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1   = nn.Linear(self.num_features, 120) # an affine operation: y = Wx + b
        self.bn1   = nn.BatchNorm1d(120)
        self.fc2   = nn.Linear(120, 84)
        self.bn2   = nn.BatchNorm1d(84)
        self.fc3   = nn.Linear(84, self.action_size)

    def zero_(self, batch_size=None):
        pass

    def forward(self, x, advice=None):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1)
        return x

class ConvNet(Module):
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

    def zero_(self, batch_size=None):
        pass

    def forward(self, *args):
        return self.neural_net.forward(*args)

class RecurrentConvNet(Module):
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


class RecurrentNet(Module):
    def __init__(self, hidden_size, *args, **kwargs):

        self.bn = kwargs.pop('bn', False)
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        input_size = self.num_features
        if self.n_friends:
            input_size += self.n_friends * C.NUM_BASIC_ACTIONS

        fc1_dim = 100
        self.fc1   = nn.Linear(input_size, fc1_dim) # an affine operation: y = Wx + b
        if self.bn:
            self.bn1   = nn.BatchNorm1d(fc1_dim)
        #self.fc2   = nn.Linear(120, gru_input_size)
        #self.bn2   = nn.BatchNorm1d(gru_input_size)

        self.gru = nn.GRUCell(input_size=fc1_dim, hidden_size=hidden_size)

        self.fc3   = nn.Linear(hidden_size, self.action_size)


    def forward(self, x, h, advice=None):
        #if advice.data.sum() > 0: import ipdb; ipdb.set_trace()
        x = x.view(-1, self.num_flat_features(x))
        if self.n_friends:
            advice = advice.view(-1, self.num_flat_features(advice))
            x = torch.cat([x, advice], 1)

        if self.bn:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        #x = F.relu(self.bn2(self.fc2(x)))

        h = self.gru(x, h)
        # output probability distribution over actions
        output = torch.mm(F.softmax(self.fc3(h) / self.temperature), Variable(self.mask))
        return output, h


class RecurrentModel(Model):
    def __init__(self, *args):
        super().__init__(*args)
        self.h = torch.zeros(self.batch_size, self.neural_net.hidden_size)

    def zero_(self, batch_size=None):
        if batch_size is not None:
            self.h.data = torch.zeros(batch_size, self.neural_net.hidden_size)
            self.batch_size = batch_size
        self.h.data.zero_()
        self.h = Variable(self.h.data)

    def forward(self, x, advice=None):
        output, self.h = self.neural_net(x, self.h, advice)
        return output
