from .maze import MazeEnv
from .utils import tensor_from_list
import torch

class Basic(MazeEnv):
    def __init__(self, **kwargs):
        walls = tensor_from_list([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]
        ).float()

        exits = torch.zeros(walls.size()).float()
        exits[6, 3] = 1

        apples = torch.zeros(walls.size()).float()
        apples[6, 1] = 1
        apples[2, 5] = 1

        oranges = torch.zeros(walls.size()).float()
        oranges[5, 1] = 1

        pears = torch.zeros(walls.size()).float()
        pears[4, 3] = 1
        pears[4, 4] = 1

        super().__init__(walls, exits, [apples, oranges, pears], **kwargs)


class NoWalls(MazeEnv):
    def __init__(self, **kwargs):
        walls = tensor_from_list([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]
        ).float()

        exits = torch.zeros(walls.size()).float()
        #exits[6, 3] = 1

        apples = torch.zeros(walls.size()).float()
        apples[6, 1] = 1
        apples[2, 5] = 1

        oranges = torch.zeros(walls.size()).float()
        oranges[5, 1] = 1

        pears = torch.zeros(walls.size()).float()
        pears[4, 3] = 1
        pears[4, 4] = 1

        super().__init__(walls, exits, [apples, oranges, pears], **kwargs)


class OneApple(MazeEnv):
    def __init__(self, **kwargs):
        walls = tensor_from_list([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]
        ).float()

        exits = torch.zeros(walls.size()).float()
        exits[6, 3] = 1

        apples = torch.zeros(walls.size()).float()
        #apples[6, 1] = 1
        apples[2, 5] = 1

        oranges = torch.zeros(walls.size()).float()
        #oranges[5, 1] = 1

        pears = torch.zeros(walls.size()).float()
        #pears[4, 3] = 1
        #pears[4, 4] = 1

        super().__init__(walls, exits, [apples, oranges, pears], **kwargs)


class ApplesEverywhere(MazeEnv):
    def __init__(self, **kwargs):
        walls = tensor_from_list([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]
        ).float()

        exits = torch.zeros(walls.size()).float()
        exits[6, 3] = 1

        apples = torch.ones(walls.size()) - walls
        oranges = torch.zeros(walls.size()).float()
        pears = torch.zeros(walls.size()).float()

        super().__init__(walls, exits, [apples, oranges, pears], **kwargs)


class EasyExit(MazeEnv):
    def __init__(self, **kwargs):
        walls = tensor_from_list([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]
        ).float()

        exits = torch.ones(walls.size()).float()

        apples = torch.zeros(walls.size()).float()
        apples[6, 1] = 1
        #apples[2, 5] = 1

        oranges = torch.zeros(walls.size()).float()
        #oranges[5, 1] = 1

        pears = torch.zeros(walls.size()).float()
        #pears[4, 3] = 1
        #pears[4, 4] = 1

        super().__init__(walls, exits, [apples, oranges, pears], **kwargs)


class RandomFruit(MazeEnv):
    def __init__(self, **kwargs):
        walls = tensor_from_list([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]
        ).float()

        exits = torch.zeros(walls.size()).float()

        apples = torch.zeros(walls.size()).float()
        oranges = torch.zeros(walls.size()).float()
        pears = torch.zeros(walls.size()).float()

        random_items = [11, 2, 5]
        super().__init__(walls, exits, [apples, oranges, pears], random_items=random_items, **kwargs)
