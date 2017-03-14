

class Constants:

    #LEARNING_RATE=0.01
    STDV=0.01
    #BATCHES=100
    MODEL_DIR='models/'
    WORKING_DIR = ''

    NUM_BASIC_ACTIONS = 7
    EPISODE_LENGTH = 50

    TIME_INCENTIVE = -0.0
    BUMP= -0.1
    MOVE= 0.
    REST= -0.1
    EMPTY_HANDED= -0
    APPLE= 2
    ORANGE= 5
    PEAR= 10
    QUIT= 1
    CALL_COST=-0.0

    COORDS = None
    TRAIN_TEMPERATURE = 10
    CONSTANT_ADVICE = True



    COLLECTED_CONSTANTS = locals()
