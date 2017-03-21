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

from constants import MazeConstants as C
from constants import ExperimentConstants as EC
from model_zoo import *
from torch_rl.policies import *

class Maze:
    def __init__(self, maze_dict):
        self.maze_dict = maze_dict
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
        w = self.walls[x, y] == 0 # cannot place agent on a wall space
        e = self.exits[x, y] == 0
        return w and e

    def get_random_valid_position(self):
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.is_valid_position(x, y):
                return x, y


class World:
    def __init__(self, maze, agent, reward, random_seed=1):
        self.maze = maze
        self.agent = agent
        self.reward = reward
        self._action_space = Discrete(agent.num_actions)
        self.num_channels = agent.num_channels + maze.num_channels
        self._state_space = Box(0.0, 1.0, (self.num_channels, self.maze.height, self.maze.width))
        self.seed = random_seed
        np.random.seed(random_seed)

    def step(self, action):
        self.on_step += 1
        self.agent.act(action, self.maze)
        self.state = self.get_state()

    def get_state(self):
        state = [torch.cat([self.agent.channels, self.maze.channels], 0)]
        #self.get_distance_from_goal()
        if self.agent.advice is not None: state += [self.agent.advice]
        if self.agent.sensors is not None: state += [torch.cat([sensor.call() for sensor in self.agent.sensors], 0)]
        return state

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

    def reset(self):
        self.on_step = 0
        self.maze.reset()
        x, y = self.maze.get_random_valid_position()
        self.agent.reset(self.maze, x, y)
        if self.maze.start_position is not None:
            self.place_agent(*self.maze.start_position)
        self.state = self.get_state()

    #def initialize(self):
    #    self._action_space = Discrete(self.agent.num_actions)
    #    self._state_space = Box(0.0, 1.0, (self.num_channels, self.maze.height, self.maze.width))


    def place_agent(self, x, y):
        if self.maze.is_valid_position(x, y):
            self.agent.reset(self.maze, x, y)
        else:
            raise ValueError("Invalid position for agent!")
        self.state = self.get_state()

class Task:
    def __init__(self, reward_dict):
        self.reward_dict    = reward_dict
        self.original_reward_dict = reward_dict

    def reward(self, world):
        if world.agent.action_type == 'move':
            if world.agent.bump:
                r = self.reward_dict['bump']
            else: r = self.reward_dict['move']
        elif world.agent.action_type == 'eat':
            if world.agent.last_meal is not None:
                if world.agent.last_meal == 0:
                    r = self.reward_dict['apple']
                elif world.agent.last_meal == 1:
                    r = self.reward_dict['orange']
                elif world.agent.last_meal == 2:
                    r = self.reward_dict['pear']
            else: r = self.reward_dict['empty_handed']
        elif world.agent.action_type == 'rest':
            r = self.reward_dict['rest']
        elif world.agent.action_type == 'quit':
            if self.finished(world):
                r = self.reward_dict['quit']
            else:
                r = self.reward_dict['rest']
        elif world.agent.action_type == 'activate_sensor':
            idx = world.agent.sensor_id
            r = self.reward_dict['sensor_costs'][idx]
        r += self.reward_dict['time_incentive']
        #r += world.get_distance_from_goal()
        return r

    def reset(self, world):
        if self.reward_dict['reward_std'] and C.EXPERIMENTAL:
            self.reward_dict['apple']  = self.original_reward_dict['apple'] + np.random.randn(1).item() * self.reward_dict['reward_std']
            self.reward_dict['orange'] = self.original_reward_dict['orange'] + np.random.randn(1).item() * self.reward_dict['reward_std']
            self.reward_dict['pear']   = self.original_reward_dict['pear'] + np.random.randn(1).item() * self.reward_dict['reward_std']
        else:
            pass

    def finished(self,world):
        done = (world.maze.item_channels.sum() == 0)
        if world.maze.regenerate:
            done = False
        done = done or world.maze.exits[world.agent.x, world.agent.y]
        #done = done and (not world.agent.playing)
        return done


class Sensor:
    """An object that can provide information about the environment to an agent"""
    def __init__(self, shape, name):
        self.shape = shape
        self.info = torch.zeros(shape)
        self.name = name


    def call(self):
        return self.info

    def zero_(self):
        self.info.zero_()

    def sense(self, state):
        pass


class RewardSensor(Sensor):
    def __init__(self, reward_key):
        super().__init__(shape=1, name="RewardSensor({})".format(reward_key))
        self.reward_key = reward_key
        self.reward = None

    def attach(self, reward):
        self.reward = reward

    def sense(self, state):
        reward_dict = self.reward.reward_dict
        self.info[0] = reward_dict[self.reward_key]


class PolicySensor(Sensor):
    def __init__(self, model_str, max_action=True):
        out_shape = C.NUM_BASIC_ACTIONS
        self.max_action = max_action
        super().__init__(shape=out_shape, name="PolicySensor({})".format(model_str))
        filename = os.path.join(EC.WORKING_DIR, 'experiments', model_str, model_str + '.ckpt')

        net = torch.load(filename)
        if net.model_type == 'ff':
            policy_model = Model(1, net, filename)
        elif net.model_type == 'recurrent':
            policy_model = RecurrentModel(1, net, filename)
        policy_model.eval()
        self.policy = DiscreteModelPolicy(Discrete(C.NUM_BASIC_ACTIONS), policy_model, allowed_actions=range(C.NUM_BASIC_ACTIONS))


    def sense(self, state):
        self.policy.observe(state)
        advice, advice_probs = self.policy.sample()
        if self.max_action:
            self.info[advice.data[0, 0]] = 1
        else:
            self.info = advice_probs.squeeze(0).data
        #import ipdb; ipdb.set_trace()


class Agent:
    def __init__(self, sensors=None):
        self.channels = None
        self.x = self.y = None
        self.direction_dict = {'down': [1, 0],
                                'up': [-1, 0],
                                'left': [0, -1],
                                'right': [0, 1]}
        self.sensors = sensors
        self.advice = None
        self.num_channels = 1
        self.num_basic_actions = C.NUM_BASIC_ACTIONS
        self.reset_num_actions()

    def assign_sensor(self, sensor):
        if self.sensors is None:
            self.sensors = [sensor]
        else: self.sensors.append(sensor)

    def reset_states(self):
        self.bump = None
        self.playing = True
        self.last_meal = None
        if self.sensors is not None:
            for sensor in self.sensors:
                sensor.zero_()

    def act(self, action, maze, constant_advice=C.CONSTANT_ADVICE):
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
            advice = self.activate_sensor(index, maze)

            if C.ACT_ON_ADVICE:
                self.act(advice, maze)
            self.action_type = 'activate_sensor'
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
                if maze.regenerate:
                    x, y = maze.get_random_valid_position()
                    channel[x, y] = 1
                self.last_meal = idx
                return idx
        self.last_meal = None
        return None

    def quit(self, maze):
        self.action_type = 'quit'
        if maze.exits[self.x, self.y]:
            self.playing = False

    def phone_friend_OLD(self, friend_id, maze):
        self.action_type = 'activate_sensor'
        self.num_calls += 1
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

    def activate_sensor(self, sensor_id, maze):
        state = torch.cat([self.position.unsqueeze(0), maze.channels], 0).unsqueeze(0)
        self.sensors[sensor_id].sense(state)
        self.sensor_id = sensor_id

    def reset_num_actions(self):
        self.num_actions = self.num_basic_actions
        if self.sensors is not None:
            self.num_actions += len(self.sensors)

    def reset(self, maze, x, y):
        self.channels = torch.zeros(self.num_channels, *maze.walls.size())
        self.position = self.channels[-1]
        assert maze.is_valid_position(x, y)
        self.position[x, y] = 1
        self.x = x
        self.y = y
        self.playing = True
        self.num_calls = 0
        self.action_type = None
        if C.EXPERIMENTAL:
            if self.sensors is not None:
                for sensor in self.sensors:
                    sensor.zero_()
            pass



class Env(gym.Env):
    '''
    A reward-based openAI Gym environment built based on a (world,reward,task) triplet
    '''
    def __init__(self,world,reward):
        self.world=world
        self.reward=reward
        self.action_space=self.world.action_space()
        self.observation_space=self.world.state_space()
        self._seed()

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
        self.reward.reset(self.world)
        return self.world.state

    def reset(self, **kwargs):
        return self._reset(**kwargs)

    #def initialize(self):
    #    self.world.initialize()
    #    self.action_space=self.world.action_space()
    #    self.observation_space=self.world.state_space()




class MazeEnv(Env):
    def __init__(self, maze_dict={}, reward_dict={},
                sensors = None,
                agent=None, world=None, reward=None, maze=None):
        if sensors is not None:
            num_sensors = len(sensors)
        else: num_sensors = 0
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
                      'sensor_costs'   : [C.SENSOR_COST] * num_sensors,
                      'reward_std'     : C.REWARD_STD
        }

        for k, v in default_reward_dict.items():
            if k not in reward_dict:
                reward_dict[k] = default_reward_dict[k]

        if maze is None: maze = Maze(maze_dict)
        if reward is None: reward = Task(reward_dict)

        if sensors is not None:
            for sensor in sensors:
                if isinstance(sensor, RewardSensor):
                    sensor.attach(reward)
        if agent is None: agent = Agent(sensors)
        if world is None: world = World(maze, agent, reward)


        self.size = (world.num_channels, maze.width, maze.height)
        super().__init__(world, reward)
