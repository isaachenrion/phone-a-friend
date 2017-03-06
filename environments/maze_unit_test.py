from maze import *

class UnitTest:
    def __init__(self):
        pass

    def test1(self):
        """Test to check if the agent can walk to an apple, eat it,
        receive correct reward and check that the apple vanishes"""
        env = MazeEnv1()
        _ = env.reset()
        env.world.place_agent(6, 6)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(2)
        #import ipdb; ipdb.set_trace()
        state, reward, finished, _ = env.step(4)
        assert reward == env.reward.apple
        assert state[0, 2, 5] == 1
        assert state[2, 2, 5] == 0
        state, reward, finished, _ = env.step(4)
        assert reward == env.reward.empty_handed

    def test2(self):
        """Test to check that the agent cannot walk through walls, and
        is punished accordingly"""
        env = MazeEnv1()
        _ = env.reset()
        env.world.place_agent(6, 6)
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.move
        state, reward, finished, _ = env.step(0)
        assert reward == env.reward.bump
        assert state[0, 1, 6] == 1

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
        assert state[2, 2, 5] == 0
        _ = env.reset()
        env.world.place_agent(6, 6)
        state, reward, finished, _ = env.step(0)
        assert state[2, 2, 5] == 1
        state, reward, finished, _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(2)
        state, reward, finished, _ = env.step(4)
        assert reward == env.reward.apple

    def test4(self):
        """ Test to check that the episode finishes when the agent collects all
        the fruit"""
        env = MazeEnv1()
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
        env = MazeEnv1()
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

    def run_all(self):
        self.test1()
        self.test2()
        self.test3()
        self.test4()
        self.test5()

ut = UnitTest()
ut.run_all()
