import torch
import numpy as np
import copy
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import collections
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

        self.state_size_dict = collections.OrderedDict()
        self.state_size_dict["spatial"] = [1 + self.maze.num_channels, self.maze.height, self.maze.width]
        if self.agent.sensors is not None:
            for s in self.agent.sensors:
                self.state_size_dict[s.name] = s.shape

        self.state = collections.OrderedDict()
        for key in self.state_size_dict.keys():
            self.state[key] = None

    def step(self, action):
        self.on_step += 1
        self.agent.act(action, self.maze)
        self.process_state()

    def process_state(self):
        self.state["spatial"] = torch.cat([self.agent.channels, self.maze.channels], 0)
        if self.agent.sensors is not None:
            for s in self.agent.sensors:
                self.state[s.name] = s.call()
        self.agent.set_state(self.state)

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
        self.process_state()

    #def initialize(self):
    #    self._action_space = Discrete(self.agent.num_actions)
    #    self._state_space = Box(0.0, 1.0, (self.num_channels, self.maze.height, self.maze.width))


    def place_agent(self, x, y):
        if self.maze.is_valid_position(x, y):
            self.agent.reset(self.maze, x, y)
        else:
            raise ValueError("Invalid position for agent!")
        self.process_state()

class Task:
    def __init__(self, reward_dict):
        self.reward_dict    = reward_dict
        self.original_reward_dict = reward_dict

    def reward(self, world):
        if world.agent.action_type in ['left', 'right', 'up', 'down']:
            if world.agent.bump:
                r = self.reward_dict['bump']
            else: r = self.reward_dict['move']
        elif world.agent.action_type == 'plus':
            if world.agent.last_item is not None:
                if world.agent.last_item == 0:
                    r = self.reward_dict['apple']
                elif world.agent.last_item == 1:
                    r = self.reward_dict['orange']
                elif world.agent.last_item == 2:
                    r = self.reward_dict['pear']
            else: r = self.reward_dict['empty_handed']
        elif world.agent.action_type == 'minus':
            if world.agent.last_item is not None:
                if world.agent.last_item == 0:
                    r = -self.reward_dict['apple']
                elif world.agent.last_item == 1:
                    r = -self.reward_dict['orange']
                elif world.agent.last_item == 2:
                    r = -self.reward_dict['pear']
            else: r = self.reward_dict['empty_handed']
        elif 'activate_sensor' in world.agent.action_type:
            idx = world.agent.sensor_id
            r = self.reward_dict['sensor_costs'][idx]
        r += self.reward_dict['time_incentive']
        #r += world.get_distance_from_goal()
        return r

    def reset(self, world):
        if self.reward_dict['reward_std'] and C.EXPERIMENTAL:
            for key in ['apple', 'pear', 'orange']:
                r = 2 * (np.random.randn(1).item() > 0) - 1
                #r  = self.original_reward_dict[key] + np.random.randn(1).item() * self.reward_dict['reward_std']
                self.reward_dict[key]  = max(min(r, +self.reward_dict['max_norm']), -self.reward_dict['max_norm'])
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
    def __init__(self, shape, name, wipe_after_call=False):
        self.shape = shape
        self.info = torch.zeros(shape)
        self.name = name
        self.wipe_after_call = wipe_after_call
        self.type=None


    def call(self):
        info = self.info.clone()
        if self.wipe_after_call:
            self.zero_()
        return info

    def zero_(self):
        self.info.zero_()

    def sense(self, state):
        pass

    def reset(self):
        self.zero_()


class RewardSensor(Sensor):
    def __init__(self, reward_key, **kwargs):
        super().__init__(shape=1, name="RewardSensor({})".format(reward_key), **kwargs)
        self.reward_key = reward_key
        self.reward = None
        self.type="reward"

    def attach(self, reward):
        self.reward = reward

    def sense(self, state):
        self.zero_()
        reward_dict = self.reward.reward_dict
        self.info[0] = reward_dict[self.reward_key]

class ActionSensor(Sensor):
    def __init__(self, action_type, **kwargs):
        super().__init__(shape=1, name="ActionSensor({})".format(action_type), **kwargs)
        self.type="action"

    def sense(self, action):
        self.zero_()
        self.info[0] = 1

class PolicySensor(Sensor):
    def __init__(self, model_str, max_action=False, **kwargs):
        out_shape = C.NUM_BASIC_ACTIONS
        self.max_action = max_action
        super().__init__(shape=out_shape, name="PolicySensor({})-{}".format(out_shape, model_str), **kwargs)
        self.type="policy"
        filename = os.path.join(EC.WORKING_DIR, 'experiments', model_str, model_str + '.ckpt')
        net = torch.load(filename)
        if net.model_type == 'ff':
            policy_model = Model(1, net, filename)
        elif net.model_type == 'recurrent':
            policy_model = RecurrentModel(1, net, filename)
        policy_model.eval()
        self.policy = DiscreteModelPolicy(Discrete(C.NUM_BASIC_ACTIONS), policy_model, allowed_actions=range(C.NUM_BASIC_ACTIONS))


    def sense(self, state):
        self.policy.observe({key: Variable(s.unsqueeze(0)) for key, s in state.items()})
        advice, advice_probs = self.policy.sample()
        self.zero_()
        if self.max_action:
            self.info[advice.data[0, 0]] = 1
        else:
            self.info = advice_probs.squeeze(0).data


class Agent:
    def __init__(self, active_sensors=None):
        self.channels = None
        self.x = self.y = None
        self.direction_dict = {'down': [1, 0],
                                'up': [-1, 0],
                                'left': [0, -1],
                                'right': [0, 1]}
        self.num_basic_actions = C.NUM_BASIC_ACTIONS
        self.reset_num_actions(active_sensors)
        self.active_sensors = active_sensors if active_sensors is not None else []
        self.num_channels = 1
        self.state = None
        self.action_types = [key for key in self.direction_dict.keys()] \
                            + ['plus', 'minus'] \
                            + ['activate_sensor_{}'.format(sensor.name) for sensor in self.active_sensors]
        self.last_action_sensors = [ActionSensor(action_type) for action_type in self.action_types]

        self.passive_sensors = self.last_action_sensors
        self.sensors = self.passive_sensors

    def reset_states(self):
        self.bump = None
        self.playing = True
        self.last_item = None

    def act(self, action, maze):
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
            self.plus(maze)
        elif action == 5:
            self.minus(maze)
        elif action >= self.num_basic_actions:
            index = action - self.num_basic_actions
            advice = self.activate_sensor(index, maze)

            #if C.ACT_ON_ADVICE:
            #    self.act(advice, maze)
        else:
            raise ValueError('Action out of bounds')
        for sensor in self.last_action_sensors:
            sensor.sense(self.action_type)
        assert self.action_type in self.action_types

    def move(self, direction_key, maze):
        self.action_type = direction_key
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

    def plus(self, maze):
        self.action_type = 'plus'
        for idx, channel in enumerate(maze.item_channels):
            if channel[self.x, self.y]:
                channel[self.x, self.y] -= 1 # clear the channels
                if maze.regenerate:
                    x, y = maze.get_random_valid_position()
                    channel[x, y] = 1
                self.last_item = idx
                return idx
        self.last_item = None
        return None

    def minus(self, maze):
        self.action_type = 'minus'
        for idx, channel in enumerate(maze.item_channels):
            if channel[self.x, self.y]:
                channel[self.x, self.y] -= 1 # clear the channels
                if maze.regenerate:
                    x, y = maze.get_random_valid_position()
                    channel[x, y] = 1
                self.last_item = idx
                return idx
        self.last_item = None
        return None

    def quit(self, maze):
        self.action_type = 'quit'
        if maze.exits[self.x, self.y]:
            self.playing = False

    def set_state(self, state):
        self.state = state

    def activate_sensor(self, sensor_id, maze):
        sensor = self.active_sensors[sensor_id]
        sensor.sense(self.state)
        self.sensor_id = sensor_id
        self.action_type = 'activate_sensor_{}'.format(sensor.name)

    def reset_num_actions(self, active_sensors):
        self.num_actions = self.num_basic_actions
        if active_sensors is not None:
            self.num_actions += len(active_sensors)

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
        if self.sensors is not None:
            for s in self.sensors:
                s.reset()




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
                active_sensors = None,
                agent=None, world=None, reward=None, maze=None):
        if active_sensors is not None:
            num_active_sensors = len(active_sensors)
        else: num_active_sensors = 0
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
                      'sensor_costs'   : [C.SENSOR_COST] * num_active_sensors,
                      'reward_std'     : C.REWARD_STD,
                      'max_norm'       : C.MAX_NORM
        }

        for k, v in default_reward_dict.items():
            if k not in reward_dict:
                reward_dict[k] = default_reward_dict[k]

        if maze is None: maze = Maze(maze_dict)
        if reward is None: reward = Task(reward_dict)

        if active_sensors is not None:
            for sensor in active_sensors:
                if isinstance(sensor, RewardSensor):
                    sensor.attach(reward)

        if agent is None: agent = Agent(active_sensors)
        if world is None: world = World(maze, agent, reward)


        self.size = (world.num_channels, maze.width, maze.height)
        super().__init__(world, reward)
