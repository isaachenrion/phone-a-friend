import torch
import numpy as np
import copy
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

import gym
from gym.utils import seeding
from gym.spaces import Discrete, Box
try:
    from .utils import tensor_from_list
except ImportError:
    from utils import tensor_from_list

from constants import Constants as C


class Maze:
    def __init__(self, maze_dict):
        self.channels = []
        self.walls = maze_dict['walls']
        self.exits = maze_dict['exits']
        self.random_items = maze_dict['random_items']
        self.random_exit = maze_dict['random_exit']
        self.start_position = maze_dict['start_position']
        self.regenerate = maze_dict['regenerate']

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
        num_actions = C.NUM_BASIC_ACTIONS
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
        #self.get_distance_from_goal()
        if self.agent.advice is not None: self.state += [self.agent.advice]

    def get_distance_from_goal(self):
        x_agent = self.agent.x
        y_agent = self.agent.y
        x, y = np.where(self.maze.exits.numpy()==1)
        x, y = x[0], y[0]
        return np.exp(-((x_agent - x) ** 2 + (y_agent - y) ** 2))

    def action_space(self):
        return self._action_space

    def state_space(self):
        return self._state_space

    def reset(self, coords=None):
        self.on_step = 0
        self.maze.reset()
        x, y = self.maze.get_random_valid_position()
        self.agent.reset(self.maze, x, y)
        if self.maze.start_position is not None:
            self.place_agent(*self.maze.start_position)
        self.state = [torch.cat([self.agent.channels, self.maze.channels], 0)]
        if self.agent.advice is not None: self.state += [self.agent.advice]

    def place_agent(self, x, y):
        if self.maze.is_valid_position(x, y):
            self.agent.reset(self.maze, x, y)
        self.state = [torch.cat([self.agent.channels, self.maze.channels], 0)]
        if self.agent.advice is not None: self.state += [self.agent.advice]


class Task:
    def __init__(self, reward_dict):
        self.reward_dict    = reward_dict
        self.time_incentive = self.reward_dict['time_incentive']
        self.bump           = self.reward_dict['bump']
        self.move           = self.reward_dict['move']
        self.rest           = self.reward_dict['rest']
        self.apple          = self.reward_dict['apple']
        self.orange         = self.reward_dict['orange']
        self.pear           = self.reward_dict['pear']
        self.empty_handed   = self.reward_dict['empty_handed']
        self.quit           = self.reward_dict['quit']
        self.call_costs     = self.reward_dict['call_costs']

    def reward(self, world):
        if world.agent.action_type == 'move':
            if world.agent.bump:
                r = self.bump
            else: r = self.move
        elif world.agent.action_type == 'eat':
            if world.agent.last_meal is not None:
                #print(world.agent.last_meal)
                if world.agent.last_meal == 0:
                    r = self.apple
                elif world.agent.last_meal == 1:
                    r = self.orange
                elif world.agent.last_meal == 2:
                    r = self.pear
            else: r = self.empty_handed
        elif world.agent.action_type == 'rest':
            r = self.rest
        elif world.agent.action_type == 'quit':
            if self.finished(world):
                r = self.quit
            else:
                r = self.rest
        elif world.agent.action_type == 'phone_friend':
            r = self.call_costs[world.agent.friend_id]
        r += self.time_incentive
        #r += world.get_distance_from_goal()
        return r


    def finished(self,world):
        done = (world.maze.item_channels.sum() == 0)
        #done = done and (not world.agent.playing)
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
        self.num_channels = 1 #+ len(self.friends)
        self.num_basic_actions = C.NUM_BASIC_ACTIONS

    def reset_states(self):
        self.bump = None
        self.playing = True
        self.last_meal = None
        if len(self.friends):
            self.advice.zero_()

    def act(self, action, maze, constant_advice=C.CONSTANT_ADVICE):
        if len(self.friends) and constant_advice:
            for idx in range(self.num_basic_actions, self.num_basic_actions + len(self.friends)):
                advice = self.phone_friend(idx - self.num_basic_actions, maze)

        self.reset_states()
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
        #elif action == 5:
        #    self.rest()
        #elif action == 6:
        #    self.quit(maze)
        elif action >= self.num_basic_actions:
            index = action - self.num_basic_actions
            advice = self.phone_friend(index, maze)

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
                x, y = maze.get_random_valid_position()
                channel[x, y] == 1
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
            advice, advice_probs = friend.sample()
            assert advice.data[0, 0] < self.num_basic_actions
            for i in range(advice_probs.size()[1]):
                self.advice[friend_id, i] = advice_probs.data[0, i]
            return advice.data[0, 0]
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
    def __init__(self, friends=[], maze_dict={}, reward_dict={}):
        self.friends = friends
        default_maze_dict = {
                        'walls'         : C.WALLS,
                        'exits'         : C.EXITS,
                        'apples'        : C.APPLES,
                        'oranges'       : C.ORANGES,
                        'pears'         : C.PEARS,
                        'regenerate'    : C.REGENERATE,
                        'random_items'  : C.RANDOM_ITEMS,
                        'random_exit'   : C.RANDOM_EXIT,
                        'start_position': C.START_POSITION
        }

        for k, v in default_maze_dict.items():
            if k not in maze_dict:
                maze_dict[k] = default_maze_dict[k]

        default_reward_dict = {
                      'time_incentive' : C.TIME_INCENTIVE,
                      'bump'           : C.BUMP,
                      'move'           : C.MOVE,
                      'rest'           : C.REST,
                      'empty_handed'   : C.EMPTY_HANDED,
                      'apple'          : C.APPLE,
                      'orange'         : C.ORANGE,
                      'pear'           : C.PEAR,
                      'quit'           : C.QUIT,
                      'call_costs'     : [C.CALL_COST] * len(friends)
        }

        for k, v in default_reward_dict.items():
            if k not in reward_dict:
                reward_dict[k] = default_reward_dict[k]

        maze = Maze(maze_dict)
        agent = Agent(friends)
        world = World(maze, agent)
        reward = Task(reward_dict)

        self.size = (world.num_channels, maze.width, maze.height)
        super().__init__(world, reward)
