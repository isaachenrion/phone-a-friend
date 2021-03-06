
import torch
class MazeConstants:

    #### REWARDS ####
    TIME_INCENTIVE = -0.00
    BUMP= -0.1
    MOVE= 0.
    REST= -0.1
    EMPTY_HANDED= -0.1
    APPLE= 2
    ORANGE= 5
    PEAR= 10
    QUIT= 0.0
    CALL_COST=-0.1
    SENSOR_COST = -0.1
    REWARD_STD=0.0
    MAX_NORM = 1

    #### MAZE ####
    WALLS = torch.Tensor([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]]).float()
    EXITS = torch.zeros(WALLS.size()).float()
    APPLES = torch.zeros(WALLS.size()).float()
    ORANGES = torch.zeros(WALLS.size()).float()
    PEARS = torch.zeros(WALLS.size()).float()
    ZEROS = torch.zeros(WALLS.size()).float()
    REGENERATE = False
    START_POSITION = None
    RANDOM_ITEMS = None
    RANDOM_EXIT = False

    #### WORLD ####
    NUM_BASIC_ACTIONS = 6
    CONSTANT_ADVICE = False
    ACT_ON_ADVICE = False
    EXPERIMENTAL = True
    GOAL_STATE = {"pears": ZEROS}
    COLLECTED_CONSTANTS = locals()

class ExperimentConstants:
    WORKING_DIR = ''

    EPISODE_LENGTH = 50
    #### TRAINING ####
    TRAIN_TEMPERATURE = 1
    EVAL_TEMPERATURE = 1

    COLLECTED_CONSTANTS = locals()
