import torch
import numpy as np
import copy

import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box
from .utils import tensor_from_list

from constants import *




class Maze:
    def __init__(self, walls, exits, item_channels, random_items=None):
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
        self.num_channels = self.channels.size()[0]
        self.random_items = random_items

    def reset(self):
        self.channels.copy_(self.original_state)
        if self.random_items is not None:
            self.item_channels.zero_()
            for channel, count in zip(self.item_channels, self.random_items):
                x, y = self.get_random_valid_position()
                channel[x, y] += 1

    def is_valid_position(self, x, y):
        return self.walls[x, y] == 0 # cannot place agent on a wall space

    def get_random_valid_position(self):
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.is_valid_position(x, y):
                return x, y
class World:
    def __init__(self, maze, agent, random_seed=1):
        self.maze = maze
        self.agent = agent
        num_actions = 4 # directions
        num_actions += 1 # eat
        num_actions += 1 # rest
        num_actions += 1 # quit
        num_actions += len(agent.friends) # friends to call
        self._action_space = Discrete(num_actions)
        self.num_channels = agent.num_channels + maze.num_channels
        self._state_space = Box(0.0, 1.0, (self.num_channels, self.maze.height, self.maze.width))
        self.seed = random_seed
        np.random.seed(random_seed)

    def step(self, action):
        self.on_step += 1
        self.agent.act(action, self.maze)
        self.state = [torch.cat([self.agent.channels, self.maze.channels], 0)]
        if self.agent.advice is not None: self.state += [self.agent.advice]

    def action_space(self):
        return self._action_space

    def state_space(self):
        return self._state_space

    def reset(self, coords=None):
        self.on_step = 0
        self.maze.reset()
        x, y = self.maze.get_random_valid_position()
        self.agent.reset(self.maze, x, y)
        if coords is not None:
            self.place_agent(*coords)
        self.state = [torch.cat([self.agent.channels, self.maze.channels], 0)]
        if self.agent.advice is not None: self.state += [self.agent.advice]

    def place_agent(self, x, y):
        if self.maze.is_valid_position(x, y):
            self.agent.reset(self.maze, x, y)
        self.state = [torch.cat([self.agent.channels, self.maze.channels], 0)]
        if self.agent.advice is not None: self.state += [self.agent.advice]


class Task:
    def __init__(self, bump, move, rest, empty_handed, apple, orange, pear, quit, call_costs):
        self.bump=bump
        self.move=move
        self.rest=rest
        self.apple=apple
        self.orange=orange
        self.pear=pear
        self.empty_handed=empty_handed
        self.quit=quit
        self.call_costs=call_costs

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
        elif world.agent.action_type == 'rest':
            return self.rest
        elif world.agent.action_type == 'quit':
            if self.finished(world):
                return self.quit
            else:
                return self.rest
        elif world.agent.action_type == 'phone_friend':
            return self.call_costs[world.agent.friend_id]


    def finished(self,world):
        done = ((world.maze.item_channels.sum() == 0) or \
                (not world.agent.playing))
        return done


class Agent:
    def __init__(self, friends=[]):
        self.channels = None
        self.x = self.y = None
        self.direction_dict = {'down': [1, 0],
                                'up': [-1, 0],
                                'left': [0, -1],
                                'right': [0, 1]}
        self.friends = friends
        self.advice = None
        self.num_channels = 1 + len(self.friends)
        self.num_basic_actions = NUM_BASIC_ACTIONS

    def reset_states(self):
        self.bump = None
        self.playing = True
        self.last_meal = None

    def act(self, action, maze):
        if len(self.friends):
            self.advice.fill_(0)
        if action == 0:
            self.move('up', maze)
        elif action == 1:
            self.move('down', maze)
        elif action == 2:
            self.move('left', maze)
        elif action == 3:
            self.move('right', maze)
        elif action == 4:
            self.eat(maze)
        elif action == 5:
            self.rest()
        elif action == 6:
            self.quit(maze)
        elif action > 6:
            index = action - 7
            advice = self.phone_friend(index, maze)
            self.act(advice, maze)
            self.action_type = 'phone_friend'
        else:
            raise ValueError('Action out of bounds')

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
                channel[self.x, self.y] -= 1 # clear the channels
                self.last_meal = idx
                return idx
        self.last_meal = None
        return None

    def quit(self, maze):
        self.action_type = 'quit'
        if maze.exits[self.x, self.y]:
            self.playing = False

    def phone_friend(self, friend_id, maze):
        self.action_type = 'phone_friend'
        self.num_calls += 1
        #print(self.action_type)
        try:
            state = torch.cat([self.position.unsqueeze(0), maze.channels], 0).unsqueeze(0)
            friend = self.friends[friend_id]
            friend.observe(state)
            self.friend_id = friend_id
            advice = friend.sample().data[0, 0]
            assert advice < self.num_basic_actions
            self.advice[friend_id, advice] = 1
            return advice
        except TypeError:
            pass

    def reset(self, maze, x, y):
        self.channels = torch.zeros(self.num_channels, *maze.walls.size())
        self.position = self.channels[-1]
        if len(self.friends):
            self.advice = torch.zeros(len(self.friends), self.num_basic_actions)
        assert maze.is_valid_position(x, y)
        self.position[x, y] = 1
        self.x = x
        self.y = y
        self.playing = True
        self.num_calls = 0


class Env(gym.Env):
    '''
    A reward-based openAI Gym environment built based on a (world,reward,task) triplet
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

    def _reset(self, **kwargs):
        self.world.reset(**kwargs)
        return self.world.state

    def reset(self, **kwargs):
        return self._reset(**kwargs)



class MazeEnv(Env):
    def __init__(self, walls, exits, item_channels, random_items=None, friends=[]):
        maze = Maze(walls, exits, item_channels, random_items=random_items)
        agent = Agent(friends)
        world = World(maze, agent)
        reward = Task(bump= BUMP,
                      move= MOVE,
                      rest= REST,
                      empty_handed= EMPTY_HANDED,
                      apple=APPLE,
                      orange=ORANGE,
                      pear= PEAR,
                      quit= QUIT,
                      call_costs=[CALL_COST] * len(friends))
        self.size = (world.num_channels, maze.width, maze.height)
        super().__init__(world, reward)
