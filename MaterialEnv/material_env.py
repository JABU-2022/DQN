import gym
from gym import spaces
import numpy as np

class MaterialEnv(gym.Env):
    def __init__(self):
        super(MaterialEnv, self).__init__()
        self.grid_size = 5
        self.agent_pos = [0, 0]
        self.materials = ['Cotton', 'Polyester', 'Wool', 'Nylon']
        self.materials_pos = {
            'Cotton': [1, 1],
            'Polyester': [1, 3],
            'Wool': [3, 1],
            'Nylon': [3, 3]
        }
        self.sustainability_scores = {
            'Cotton': 10,
            'Polyester': 2,
            'Wool': 7,
            'Nylon': 3
        }
        self.max_steps = 50
        self.current_step = 0

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)

    def reset(self):
        self.agent_pos = [0, 0]
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.agent_pos)

    def step(self, action):
        # Update agent's position based on action
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

        # Calculate reward
        reward = -0.01  # Small penalty for each step to encourage movement
        for material, pos in self.materials_pos.items():
            if self.agent_pos == pos:
                reward = self.sustainability_scores[material]
                break

        done = False
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), ' ')
        for material, pos in self.materials_pos.items():
            grid[pos[0], pos[1]] = material[0]
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        print(grid)
