import sys, os
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))
from maze import *
from maze_envs import *

class UnitTest:
    def __init__(self):
        pass

    def test1(self):
        """Test to check if the agent can walk to an apple, eat it,
        receive correct reward and check that the apple vanishes"""
        env = Basic()
        _ = env.reset()
        env.world.place_agent(6, 6)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(2)

        #import ipdb; ipdb.set_trace()
        state, reward, finished, _ = env.step(4)
        assert reward == env.reward.apple + env.reward.time_incentive
        assert state[0][0, 2, 5] == 1
        assert state[0][2, 2, 5] == 0
        state, reward, finished, _ = env.step(4)
        assert reward == env.reward.empty_handed + env.reward.time_incentive

    def test2(self):
        """Test to check that the agent cannot walk through walls, and
        is punished accordingly"""
        env = Basic()
        _ = env.reset()
        env.world.place_agent(6, 6)
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move + env.reward.time_incentive
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move + env.reward.time_incentive
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move + env.reward.time_incentive
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move + env.reward.time_incentive
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move + env.reward.time_incentive
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.bump + env.reward.time_incentive
        assert state[0][0, 1, 6] == 1

    def test3(self):
        """ Test to check that the world resets properly and fruits regenerate
        """
        env = OneApple()
        _ = env.reset()
        env.world.place_agent(6, 6)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(2)
        state, reward, finished, _ = env.step(4)
        assert finished
        assert state[0][2, 2, 5] == 0
        _ = env.reset()
        env.world.place_agent(6, 6)
        state, reward, finished, _ = env.step(0)
        assert state[0][2, 2, 5] == 1
        state, reward, finished, _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(2)
        state, reward, finished, _ = env.step(4)
        assert reward == env.reward.apple + env.reward.time_incentive

    def test4(self):
        """ Test to check that the episode finishes when the agent collects all
        the fruit"""
        env = Basic()
        _ = env.reset()
        env.world.place_agent(6, 6)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(2)
        _ = env.step(4)
        _ = env.step(3)
        _ = env.step(1)
        _ = env.step(1)
        _ = env.step(2)
        _ = env.step(2)
        _ = env.step(4)
        _ = env.step(2)
        _ = env.step(4)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(2)
        _ = env.step(2)
        _ = env.step(1)
        _ = env.step(1)
        _ = env.step(1)
        _ = env.step(1)
        _ = env.step(4)
        _ = env.step(1)
        state, reward, finished, _ = env.step(4)
        assert finished

    def test5(self):
        """Test to check that the quit option works at an exit, and doesn't work
        otherwise"""
        env = OneApple()
        _ = env.reset()
        env.world.place_agent(6, 6)
        _ = env.step(0)
        _ = env.step(0)
        state, reward, finished, _ = env.step(6)
        assert not finished
        _ = env.step(2)
        _ = env.step(2)
        _ = env.step(2)
        _ = env.step(1)
        _ = env.step(1)
        state, reward, finished, _ = env.step(6)
        assert finished

    def test6(self):
        """Test to check that advice from a friend is integrated correctly into
        the world state"""
        from model_zoo import Model
        from torch_rl.policies import DiscreteModelPolicy
        from gym.spaces import Discrete
        from constants import Constants as C
        model_strs = []
        model_strs.append('Mar-14___15-23-34-RandomPear.ckpt')
        friends = []
        for model_str in model_strs:
            filename = os.path.join('..', C.MODEL_DIR, model_str)
            friend_model = Model(1, torch.load(filename), filename)
            friend_model.eval()
            friend = DiscreteModelPolicy(Discrete(C.NUM_BASIC_ACTIONS), friend_model, disallowed_actions=[act for act in range(C.NUM_BASIC_ACTIONS, C.NUM_BASIC_ACTIONS+len(model_strs))])
            friends.append(friend)
        env = RandomPear(friends=friends)
        state = env.reset()
        import ipdb; ipdb.set_trace()


    def run_all(self):
        self.test1()
        self.test2()
        self.test3()
        self.test4()
        self.test5()

ut = UnitTest()
#ut.run_all()
ut.test6()
