
import torch
class MazeConstants:

    #### REWARDS ####
    TIME_INCENTIVE = -0.0
    BUMP= -0.1
    MOVE= 0.
    REST= -0.1
    EMPTY_HANDED= -0.1
    APPLE= 2
    ORANGE= 5
    PEAR= 10
    QUIT= 1
    CALL_COST=-0.1
    SENSOR_COST = -0.1
    REWARD_STD=0.0

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
    EXITS[6, 1] = 1
    APPLES = torch.zeros(WALLS.size()).float()
    ORANGES = torch.zeros(WALLS.size()).float()
    PEARS = torch.zeros(WALLS.size()).float()
    REGENERATE = False
    START_POSITION = None
    RANDOM_ITEMS = None
    RANDOM_EXIT = False

    #### WORLD ####
    NUM_BASIC_ACTIONS = 5
    CONSTANT_ADVICE = False
    ACT_ON_ADVICE = False
    EXPERIMENTAL = True

    COLLECTED_CONSTANTS = locals()

class ExperimentConstants:
    WORKING_DIR = ''

    EPISODE_LENGTH = 50
    #### TRAINING ####
    TRAIN_TEMPERATURE = 1
    EVAL_TEMPERATURE = 1

    COLLECTED_CONSTANTS = locals()
