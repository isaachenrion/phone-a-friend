import torch
import numpy as np
import copy

import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box
from .utils import tensor_from_list

class Maze:
    def __init__(self, walls, exits, *item_channels):
        self.channels = []
        self.walls = walls
        self.exits = exits
        self.channels.append(self.walls)
        for channel in item_channels:
            assert channel.size() == walls.size()
            assert (channel * walls).sum() == 0 # no fruit in walls
            self.channels.append(channel)
        self.channels.append(exits)
        self.height, self.width = self.walls.size()
        self.num_items = len(item_channels)
        self.channels = torch.stack(self.channels, 0)
        self.item_channels = self.channels[1:-1]
        self.original_state = self.channels.clone()

    def reset(self):
        self.channels.copy_(self.original_state)


class World:
    def __init__(self, maze, agent, random_seed=1):
        self.maze = maze
        self.agent = agent
        self._action_space = Discrete(7)
        self.num_channels = 1 + 1 + maze.num_items + 1
        self._state_space = Box(0.0, 1.0, (self.num_channels, self.maze.height, self.maze.width))
        self.seed = random_seed
        np.random.seed(random_seed)

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
            self.agent.eat(self.maze)
        elif action == 5:
            self.agent.rest()
        elif action == 6:
            self.agent.quit(self.maze)
        else:
            raise ValueError('Action out of bounds')
        self.state = torch.cat([self.agent.position.unsqueeze(0), self.maze.channels], 0)

    def action_space(self):
        return self._action_space

    def state_space(self):
        return self._state_space

    def reset(self):
        self.on_step = 0
        self.maze.reset()
        while True:
            x = np.random.randint(0, self.maze.width)
            y = np.random.randint(0, self.maze.height)
            #print('{}, {}'.format(x, y))
            if self.agent.is_valid_position(self.maze, x, y):
                self.agent.reset(self.maze, x, y)
                break
        self.state = torch.cat([self.agent.position.unsqueeze(0), self.maze.channels], 0)

    def place_agent(self, x, y):
        if self.agent.is_valid_position(self.maze, x, y):
            self.agent.reset(self.maze, x, y)
        self.state = torch.cat([self.agent.position.unsqueeze(0), self.maze.channels], 0)


class Task:
    def __init__(self, bump=-2, move=-0.1, rest=0, empty_handed=-1, apple=5, orange=20, pear=10):
        self.bump=bump
        self.move=move
        self.rest=rest
        self.apple=apple
        self.orange=orange
        self.pear=pear
        self.empty_handed=empty_handed
        self.quit=0

    def reward(self, world):
        print(world.agent.action_type)
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
        elif world.agent.action_type == 'rest':
            return self.rest
        elif world.agent.action_type == 'quit':
            if world.agent.playing:
                return self.rest
            else:
                return self.quit


    def finished(self,world):
        done = ((world.maze.item_channels.sum() == 0) and \
                (not world.agent.playing))
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

    def rest(self):
        self.action_type = 'rest'

    def eat(self, maze):
        self.action_type = 'eat'
        for idx, channel in enumerate(maze.item_channels):
            if channel[self.x, self.y]:
                channel[self.x, self.y] = 0 # clear the position
                self.last_meal = idx
                return idx
        self.last_meal = None
        return None

    def quit(self, maze):
        self.action_type = 'quit'
        #print("trying to quit")
        if maze.exits[self.x, self.y]:
            self.playing = False
            #print("quit")

    def is_valid_position(self, maze, x, y):
        return maze.walls[x, y] == 0 # cannot place agent on a wall space

    def reset(self, maze, x, y):
        assert self.is_valid_position(maze, x, y)
        self.position = torch.zeros(maze.walls.size())
        self.position[x, y] = 1
        self.x = x
        self.y = y
        self.playing = True


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


class MazeEnv(Env):
    def __init__(self, walls, exits, apples, oranges, pears):
        maze = Maze(walls, exits, apples, oranges, pears)
        agent = Agent()
        world = World(maze, agent)
        reward = Task(bump= -5,
                      move= -1,
                      rest= -1,
                      empty_handed= -2,
                      apple= 0,
                      orange= 0,
                      pear= 0)
        self.size = (world.num_channels, maze.width, maze.height)
        super().__init__(world, reward)


class MazeEnv1(MazeEnv):
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

        super().__init__(walls, exits, apples, oranges, pears)


class OneApple(MazeEnv):
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

        super().__init__(walls, exits, apples, oranges, pears)
