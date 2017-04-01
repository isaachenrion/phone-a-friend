from .maze import Maze
from .agent import Agent, Subordinate
from .sensor import *
from .task import Task
from .world import World
from .env import Env

from constants import MazeConstants as C
from constants import ExperimentConstants as EC
class MazeEnv(Env):
    def __init__(self, maze_dict={}, reward_dict={},
                active_sensors = None, subordinates = None,
                agent=None, world=None, reward=None, maze=None,
                seed=None):
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
                      'max_norm'       : C.MAX_NORM,
                      'goal_state'     : C.GOAL_STATE
        }

        for k, v in default_reward_dict.items():
            if k not in reward_dict:
                reward_dict[k] = default_reward_dict[k]

        if maze is None: maze = Maze(maze_dict, seed)
        if reward is None: reward = Task(reward_dict)

        if active_sensors is not None:
            for sensor in active_sensors:
                if isinstance(sensor, RewardSensor):
                    sensor.attach(reward)

        if agent is None: agent = Agent(active_sensors, subordinates)
        if world is None: world = World(maze, agent, reward)


        self.size = (world.num_channels, maze.width, maze.height)
        hierarchical = subordinates is not None
        #import ipdb; ipdb.set_trace()
        #print(seed)
        super().__init__(world, reward, hierarchical)
        #print(self.np_random)
