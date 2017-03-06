import torch
import numpy as np

def tensor_from_list(list_):
    return torch.from_numpy(np.array(list_))
