import torch
import numpy as np
from torch.autograd import Variable

def tensor_from_list(list_):
    return torch.from_numpy(np.array(list_))

def unsqueeze_dict(tensor_dict):
    return {k : v.unsqueeze(0) for k, v in tensor_dict.items()}

def variablize(tensor_dict):
    return {k : Variable(v.unsqueeze(0)) for k, v in tensor_dict.items()}
