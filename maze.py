import torch
import numpy as np
from utils import tensor_from_list
import copy

import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box


class Maze:
    def __init__(self, walls, *item_channels):
        self.channels = []
        self.walls = walls
        self.channels.append(self.walls)
        for channel in item_channels:
            assert channel.size() == walls.size()
            assert (channel * walls).sum() == 0 # no fruit in walls
            self.channels.append(channel)
        self.height, self.width = self.walls.size()
        self.num_items = len(item_channels)
        self.channels = torch.stack(self.channels, 0)
        self.item_channels = self.channels[1:]

class World:
    def __init__(self, maze, agent):
        self.maze = maze
        self.original_maze = copy.deepcopy(maze)
        self.agent = agent
        self._action_space = Discrete(4 + 1)
        self._state_space = Box(0.0, 1.0, (1 + 1 + maze.num_items, self.maze.height, self.maze.width))

    def init_agent(self, x, y):
        assert self.maze.walls[x, y] == 0 # cannot place agent on a wall space
        self.agent.position = torch.zeros(self.maze.walls.size())
        self.agent.position[x, y] = 1
        self.agent.x = x
        self.agent.y = y

    def step(self, action):
        self.on_step += 1
        if action == 0:
            self.agent.move('up', self.maze)
        elif action == 1:
            self.agent.move('down', self.maze)
        elif action == 2:
            self.agent.move('left', self.maze)
        elif action == 3:
            self.agent.move('right', self.maze)
        elif action == 4:
            self.agent.grab(self.maze)
        else:
            raise ValueError('Action out of bounds')
        self.state = torch.cat([self.agent.position.unsqueeze(0), self.maze.channels], 0)


    def action_space(self):
        return self._action_space

    def state_space(self):
        return self._state_space

    def reset(self):
        self.on_step = 0
        self.init_agent(6, 6)
        self.state = torch.cat([self.agent.position.unsqueeze(0), self.maze.channels], 0)

class Task:
    def __init__(self, bump=-2, move=0, empty_handed=-1, apple=5, orange=20, pear=-10):
        self.bump=bump
        self.move=move
        self.apple=apple
        self.orange=orange
        self.pear=pear
        self.empty_handed=empty_handed

    def reward(self, world):
        if world.agent.action_type == 'move':
            if world.agent.bump:
                return self.bump
            else: return self.move
        elif world.agent.action_type == 'eat':
            if world.agent.last_meal is not None:
                if world.agent.last_meal == 0:
                    return self.apple
                elif world.agent.last_meal == 1:
                    return self.orange
                elif world.agent.last_meal == 2:
                    return self.pear
            else: return self.empty_handed

    def finished(self,world):
        done = (world.on_step == 50)
        return done


class Agent:
    def __init__(self):
        self.position = None
        self.x = self.y = None
        self.direction_dict = {'down': [1, 0],
                                'up': [-1, 0],
                                'left': [0, -1],
                                'right': [0, 1]}

    def move(self, direction_key, maze):
        self.action_type = 'move'
        direction = self.direction_dict[direction_key]
        candidate_x = self.x + direction[0]
        candidate_y = self.y + direction[1]
        if maze.walls[candidate_x, candidate_y]:
            self.bump = True
        else:
            self.bump = False
            self.position[self.x, self.y] = 0
            self.position[candidate_x, candidate_y] = 1
            self.x = candidate_x
            self.y = candidate_y

    def eat(self, maze):

        self.action_type = 'eat'
        for idx, channel in enumerate(maze.item_channels):
            if channel[self.x, self.y]:
                channel[self.x, self.y] = 0 # clear the position
                self.last_meal = idx
                return idx
        self.last_meal = None
        return None

class Env(gym.Env):
    '''
    A reward-based oepnAI Gym environment built based on a (world,reward,task) triplet
    '''
    def __init__(self,world,reward):
        self.world=world
        self.reward=reward
        self._seed()
        self.action_space=self.world.action_space()
        self.observation_space=self.world.state_space()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.world.step(action)
        immediate_reward=self.reward.reward(self.world)
        observation=self.world.state
        finished=self.reward.finished(self.world)
        return observation,immediate_reward,finished,None

    def _reset(self):
        self.world.reset()
        return self.world.state


class MazeEnv1(Env):
    def __init__(self):
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

        apples = torch.zeros(walls.size()).float()
        apples[6, 1] = 1
        apples[2, 5] = 1

        oranges = torch.zeros(walls.size()).float()
        oranges[5, 1] = 1

        pears = torch.zeros(walls.size()).float()
        pears[4, 3] = 1
        pears[4, 4] = 1

        maze = Maze(walls, apples, oranges, pears)
        agent = Agent()
        world = World(maze, agent)
        reward = Task()
        super().__init__(world, reward)
