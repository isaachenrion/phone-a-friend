import gym
from gym.utils import seeding
class Env(gym.Env):
    '''
    A reward-based openAI Gym environment built based on a (world,reward,task) triplet
    '''
    def __init__(self,world,reward, hierarchical=False):
        self.world=world
        self.reward=reward
        self.action_space=self.world.action_space()
        self.observation_space=self.world.state_space()
        self._seed()
        self.hierarchical = hierarchical

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step_vanilla(self, action):
        self.world.step(action)
        immediate_reward=self.reward.reward(self.world)
        observation=self.world.state
        finished=self.reward.finished(self.world)
        return observation,immediate_reward,finished,None

    def _step_hierarchical(self, action):
        if self.world.agent.current_subordinate is None:
            self.world.step(action)
            immediate_reward=self.reward.reward(self.world)
            observation=self.world.state
            finished=self.reward.finished(self.world)
            return observation,immediate_reward,finished,None
        else:
            subordinate = self.world.agent.current_subordinate
            observation,r,finished,_ = subordinate.operate(self.world, self.reward)
            return observation,r,finished,None

    def _step(self, action):
        if self.hierarchical and False:
            return self._step_hierarchical(action)
        else:
            return self._step_vanilla(action)

    def _reset(self, **kwargs):
        self.world.reset(**kwargs)
        self.reward.reset(self.world)
        return self.world.state

    def reset(self, **kwargs):
        return self._reset(**kwargs)
