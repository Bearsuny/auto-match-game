import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from auto_match_game.ui.auto_match_game_ui import AutoMatchGameUI


class AutoMatchGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.ui = AutoMatchGameUI()
        self.ui.show()
        self.ui.train_button.clicked.connect(self.train)

    def step(self, action):
        observation, reward, done, info = 0, 0, False, 0
        return observation, reward, done, info

    def reset(self):
        self.ui.reset()

    def render(self, mode='human'):
        pass

    def train(self):
        epochs = 1000
        time_steps = 1000
        for epoch in range(epochs):
            self.reset()
            for time_step in range(time_steps):
                self.render()
                action = None
                observation, reward, done, info = self.step(action)
                if time_step == 100:
                    done = True
                if done:
                    print(f'Epoch: {epoch:03}, Stop after {time_step:03} steps, Reward is {reward}.')
                    break
