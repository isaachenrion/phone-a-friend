from maze import MazeEnv1

class UnitTest:
    def __init__(self):
        pass

    def test1(self):
        """Test to check if the agent can walk to an apple, eat it,
        receive correct reward and check that the apple vanishes"""
        env = MazeEnv1()
        _ = env.reset()
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(2)
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

ut = UnitTest()
ut.test1()
ut.test2()
