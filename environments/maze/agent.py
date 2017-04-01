import torch
import numpy as np
import sys, os

from .sensor import ActionSensor
from .utils import variablize
from constants import MazeConstants as C
from constants import ExperimentConstants as EC
from model_zoo import RecurrentModel
from torch_rl.policies import DiscreteModelPolicy
from gym.spaces import Discrete

class Agent:
    def __init__(self, active_sensors=None, subordinates=None):
        self.state = None
        self.x = None
        self.y = None
        self.direction_dict = {'down': [1, 0],
                                'up': [-1, 0],
                                'left': [0, -1],
                                'right': [0, 1]}
        self.active_sensors = active_sensors if active_sensors is not None else []
        self.subordinates = subordinates if subordinates is not None else {}
        self.action_types = [key for key in self.direction_dict.keys()] \
                            + ['plus', 'minus', 'quit'] \
                            + ['activate_sensor/{}'.format(sensor.name) for sensor in self.active_sensors]\
                            + ['activate_subordinate/{}'.format(name) for name in self.subordinates]
        self.num_actions = len(self.action_types)
        self.last_action_sensors = [ActionSensor(action_type) for action_type in self.action_types]

        self.passive_sensors = self.last_action_sensors
        self.sensors = self.passive_sensors

    def reset_states(self):
        self.bump = None
        self.playing = True
        self.last_item = None
        self.action_type = None

    def act(self, action, maze):
        self.reset_states()
        action = self.action_types[action]
        if action == 'up':
            self.move('up', maze)
        elif action == 'down':
            self.move('down', maze)
        elif action == 'left':
            self.move('left', maze)
        elif action == 'right':
            self.move('right', maze)
        elif action == 'plus':
            self.plus(maze)
        elif action == 'minus':
            self.minus(maze)
        elif action == 'quit':
            self.quit()
        #elif 'activate_sensor' in action:
        #    index = action - self.num_basic_actions
        #    self.activate_sensor(index, maze)
        elif 'activate_subordinate' in action:
            sub_name = action.split('/')[1]
            self.activate_subordinate(sub_name)
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
            self.state["agent_spatial"][self.x, self.y] = 0
            self.state["agent_spatial"][candidate_x, candidate_y] = 1
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

    def quit(self):
        self.action_type = 'quit'
        self.playing = False

    def activate_sensor(self, sensor_id, maze):
        sensor = self.active_sensors[sensor_id]
        sensor.sense(self.state)
        self.sensor_id = sensor_id
        self.action_type = 'activate_sensor_{}'.format(sensor.name)

    def activate_subordinate(self, sub_name):
        self.action_type = 'activate_subordinate/{}'.format(sub_name)
        #self.current_subordinate = self.subordinates[sub_name]
        # seed subordinate with hidden state!

    def deactivate_subordinate(self, policy_id):
        pass

    def reset_num_actions(self, active_sensors, subordinates):
        self.num_actions = self.num_basic_actions
        if active_sensors is not None:
            self.num_actions += len(active_sensors)
        if subordinates is not None:
            self.num_actions += len(subordinates)

    def reset(self, maze, x, y):
        self.state = {"agent_spatial": torch.zeros(*maze.walls.size())}
        assert maze.is_valid_position(x, y)
        self.state["agent_spatial"][x, y] = 1
        self.x = x
        self.y = y
        self.playing = True
        self.num_calls = 0
        self.action_type = None
        if self.sensors is not None:
            for s in self.sensors:
                s.reset()
        #self.current_subordinate = None



class Subordinate:
    def __init__(self, model_str, **kwargs):
        filename = os.path.join(EC.WORKING_DIR, 'experiments', model_str, model_str + '.ckpt')
        net = torch.load(filename)
        policy_model = RecurrentModel(1, net, filename)
        policy_model.eval()
        self.policy = DiscreteModelPolicy(action_space=Discrete(C.NUM_BASIC_ACTIONS), action_model=policy_model, allowed_actions=range(C.NUM_BASIC_ACTIONS))
        self.goal = net.goal_state

    def operate(self, world, reward):
        r = 0.
        t = 0
        actions_taken=[]
        while t < 50:
            self.policy.observe(variablize(world.state))
            action, scores = self.policy.sample()
            actions_taken.append(action.data[0][0])
            world.step(action.data[0][0])
            immediate_reward=reward.reward(world)
            state_reached=reward.state_reached(world, self.goal)
            r += immediate_reward
            if state_reached: break
            t += 1
        observation=world.state
        finished = reward.finished(world)
        return observation,r,finished,None
