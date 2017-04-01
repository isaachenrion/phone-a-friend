import torch
import numpy as np
import copy
from constants import MazeConstants as C
from constants import ExperimentConstants as EC

class Task:
    def __init__(self, reward_dict):
        self.reward_dict    = reward_dict.copy()
        self.original_reward_dict = reward_dict.copy()
        self.goal_state = self.reward_dict['goal_state']

    def _reward(self, world):
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
        elif world.agent.action_type == 'quit':
            r = self.reward_dict['quit']
        elif 'activate_sensor' in world.agent.action_type:
            idx = world.agent.sensor_id
            r = self.reward_dict['sensor_costs'][idx]
        elif 'activate_subordinate' in world.agent.action_type:
            r = 0.
        r += self.reward_dict['time_incentive']
        return r

    def _reward2(self, world):
        if self.goal_state_reached(world) and not world.agent.playing:
            r = 10.
        else:
            r = -0.0
        return r

    def reward(self, world):
        return self._reward2(world)

    def reset(self, world):
        if not C.EXPERIMENTAL and self.reward_dict['reward_std']:
            for key in ['apple', 'pear', 'orange']:
                r = 2 * (np.random.randn(1).item() > 0) - 1
                self.reward_dict[key]  = max(min(r, +self.reward_dict['max_norm']), -self.reward_dict['max_norm'])
        else:
            pass

    def _finished(self,world):
        done = (world.maze.item_channels.sum() == 0)
        if world.maze.regenerate:
            done = False
        done = done or world.maze.exits[world.agent.x, world.agent.y]
        return done

    def _finished2(self, world):
        return self.goal_state_reached(world) and not world.agent.playing

    def finished(self, world):
        if C.EXPERIMENTAL:
            return self._finished2(world)
        else:
            return self._finished(world)

    def goal_state_reached(self, world):
        return self.state_reached(world, self.goal_state)

    def state_reached(self, world, state):
        done = True
        for k in state.keys():
            done = done and torch.prod(world.state[k] == state[k])
        return done
