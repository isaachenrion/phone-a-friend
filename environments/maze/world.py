from gym.spaces import Discrete, Box
import collections
import numpy as np

class World:
    def __init__(self, maze, agent, reward, random_seed=1):
        self.maze = maze
        self.agent = agent
        self.reward = reward
        self._action_space = Discrete(agent.num_actions)
        self.num_channels = 1 + maze.num_channels
        self._state_space = Box(0.0, 1.0, (self.num_channels, self.maze.height, self.maze.width))
        self.seed = random_seed
        np.random.seed(random_seed)

        self.state_size_dict = collections.OrderedDict()
        self.state_size_dict["agent_spatial"] = [self.maze.height, self.maze.width]
        #for name, sub in self.agent.subordinates.items():
        #    self.state_size_dict[name] = 1
        for name in self.maze.state.keys():
            self.state_size_dict[name]  = [self.maze.height, self.maze.width]
        if self.agent.sensors is not None:
            for s in self.agent.sensors:
                self.state_size_dict[s.name] = s.shape

        self.state = collections.OrderedDict()
        for key in self.state_size_dict.keys():
            self.state[key] = None

    def step(self, action):
        self.on_step += 1
        self.agent.act(action, self.maze)
        self.process_state()

    def process_state(self):
        for name, channel in self.agent.state.items():
            self.state[name] = channel
        for name, channel in self.maze.state.items():
            self.state[name] = channel
        if self.agent.sensors is not None:
            for s in self.agent.sensors:
                self.state[s.name] = s.call()

    def get_distance_from_goal(self):
        x_agent = self.agent.x
        y_agent = self.agent.y
        x, y = np.where(self.maze.exits.numpy()==1)
        x, y = x[0], y[0]
        return np.exp(-((x_agent - x) ** 2 + (y_agent - y) ** 2))

    def action_space(self):
        return self._action_space

    def state_space(self):
        return self._state_space

    def reset(self):
        self.on_step = 0
        self.maze.reset()
        x, y = self.maze.get_random_valid_position()
        self.agent.reset(self.maze, x, y)
        if self.maze.start_position is not None:
            self.place_agent(*self.maze.start_position)
        self.process_state()

    def place_agent(self, x, y):
        if self.maze.is_valid_position(x, y):
            self.agent.reset(self.maze, x, y)
        else:
            raise ValueError("Invalid position for agent!")
        self.process_state()
