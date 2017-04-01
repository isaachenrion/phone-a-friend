import torch
import numpy as np
import copy

from constants import MazeConstants as C
from constants import ExperimentConstants as EC
class Maze:
    def __init__(self, maze_dict, seed=None):
        self.maze_dict = maze_dict
        self.channels = []
        self.walls = maze_dict['walls']
        self.exits = maze_dict['exits']
        self.random_items = maze_dict['random_items']
        self.random_exit = maze_dict['random_exit']
        self.start_position = maze_dict['start_position']
        self.regenerate = maze_dict['regenerate']
        self.rng = np.random.RandomState(seed)

        item_channels = [maze_dict['apples'], maze_dict['oranges'], maze_dict['pears']]
        self.channels.append(self.walls)
        for channel in item_channels:
            assert channel.size() == self.walls.size()
            assert (channel * self.walls).sum() == 0 # no fruit in walls
            self.channels.append(channel)
        self.channels.append(maze_dict['exits'])
        self.height, self.width = self.walls.size()
        self.num_items = len(item_channels)
        self.channels = torch.stack(self.channels, 0)
        self.item_channels = self.channels[1:-1]
        self.original_state = self.channels.clone()
        self.num_channels = self.channels.size()[0]
        self.state = {"walls": self.walls, "exits": self.exits, "apples": self.item_channels[0], "oranges": self.item_channels[1], "pears": self.item_channels[2]}


    def reset(self):
        self.channels.copy_(self.original_state)
        if self.random_items is not None:
            self.item_channels.zero_()
            for channel, count in zip(self.item_channels, self.random_items):
                for _ in range(count):
                    x, y = self.get_random_valid_position()
                    channel[x, y] += 1
        if self.random_exit:
            self.exits = torch.zeros(self.exits.size())
            x, y = self.get_random_valid_position()
            self.exits[x, y] = 1

    def is_valid_position(self, x, y):
        w = self.walls[x, y] == 0 # cannot place agent on a wall space
        e = self.exits[x, y] == 0
        return w and e

    def get_random_valid_position(self):
        while True:
            x = self.rng.randint(0, self.width)
            y = self.rng.randint(0, self.height)
            if self.is_valid_position(x, y):
                return x, y
