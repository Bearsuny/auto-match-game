import gym
from gym import spaces
import os
import numpy as np
from PyQt5.QtGui import QPixmap
from auto_match_game.ui.config import AutoMatchGameUIConfig as UIC
from auto_match_game.envs.config import AutoMatchGameEnvConfig as EC
from reward_grid.config import RewardGridConfig as RGC

from auto_match_game.ui.auto_match_game_ui import AutoMatchGameUI
from reward_grid.net_eval import get_reward_grid_roi

import torch

from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal


class AutoMatchGameThread(QThread):
    step_signal = pyqtSignal(np.ndarray)
    reset_signal = pyqtSignal()
    render_signal = pyqtSignal()

    def __init__(self, env):
        super(AutoMatchGameThread, self).__init__()
        self.env = env
        self.observation = None
        self.reward = None
        self.done = False
        self.info = None

    def run(self):
        for i in range(EC.train_max_epochs):
            self.reset_signal.emit()
            for j in range(EC.train_max_steps):
                self.render_signal.emit()
                action = self.env.action_space.sample()
                self.step_signal.emit(action)
                QThread.msleep(500)
                if self.done:
                    print(f'Epoch {i:03}, Step {j:03} steps, Reward is {self.reward}.')
                    break
        print('Train done.')
        self.stop()

    def stop(self):
        self.quit()


class AutoMatchGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.ui = AutoMatchGameUI()
        self.ui.show()
        self.ui.train_button.clicked.connect(self.train)

        self.max_epochs = EC.train_max_epochs
        self.max_steps = EC.train_max_steps

        self.game_steps = EC.game_steps

        self.reward_grid_net = torch.load(RGC.reward_grid_net_save_path)
        self.reward_grid_net.to(self.reward_grid_net.device)
        self.reward_grid_net.eval()

        self.action_space = spaces.MultiDiscrete(np.array([EC.grid_row, EC.grid_col, EC.direction]))

        self.grid_result = []

    def train(self):
        self.ui.amg_thread = AutoMatchGameThread(self)

        self.ui.amg_thread.step_signal.connect(self.step)
        self.ui.amg_thread.reset_signal.connect(self.reset)
        self.ui.amg_thread.render_signal.connect(self.render)

        self.ui.amg_thread.start()

    def step(self, action):
        self.generate_grid_result()
        x, y, direction = action
        flag1 = False
        if action[-1] == 0:
            if x - 1 >= 0:
                flag1 = True
                self.grid_result[x - 1][y], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x - 1][y]
        if action[-1] == 1:
            if y + 1 <= EC.grid_col - 1:
                flag1 = True
                self.grid_result[x][y + 1], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x][y + 1]
        if action[-1] == 2:
            if x + 1 <= EC.grid_row - 1:
                flag1 = True
                self.grid_result[x + 1][y], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x + 1][y]
        if action[-1] == 3:
            if y - 1 >= 0:
                flag1 = True
                self.grid_result[x][y - 1], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x][y - 1]

        # if flag1:
        #     self.calc_reward()
        #     self.game_steps -= 1
        self.calc_reward()
        self.game_steps -= 1

        if self.game_steps == 0:
            self.ui.amg_thread.done = True

    def reset(self):
        self.game_steps = EC.game_steps
        self.ui.amg_thread.done = False
        for position in self.ui.positions:
            pick_num = np.random.randint(0, len(self.ui.ref_images))
            pixmap = QPixmap(os.path.join(UIC.reward_ref_image_path, self.ui.ref_images[int(pick_num)]))
            pixmap = pixmap.scaled(*UIC.scale_size)
            self.ui.label_list[position[0] * UIC.reward_row + position[1]].setPixmap(pixmap)
        self.generate_grid_result()

    def render(self, mode='human'):
        self.ui.step_label.setText(f'STEP: {self.game_steps}')
        self.ui.score_label.setText(f'SCORE: {self.ui.amg_thread.reward}')
        for position in self.ui.positions:
            pixmap = QPixmap(os.path.join(UIC.reward_ref_image_path, self.ui.ref_images[int(self.grid_result[position[0]][position[1]])]))
            pixmap = pixmap.scaled(*UIC.scale_size)
            self.ui.label_list[position[0] * UIC.reward_row + position[1]].setPixmap(pixmap)

    def generate_grid_result(self):
        self.grid_result = []
        pix = self.ui.grab()
        pix.save(UIC.screen_output_path)
        reward_grid_roi = get_reward_grid_roi(UIC.screen_output_path, UIC.crop_box)

        with torch.no_grad():
            for row_item in range(RGC.reward_grid_row):
                data = reward_grid_roi[row_item * RGC.reward_grid_col:(row_item + 1) * RGC.reward_grid_col]
                data = torch.from_numpy(data).float()
                data = data.to(self.reward_grid_net.device)
                output = self.reward_grid_net(data)
                result = torch.argmax(output, dim=1)
                result = result.cpu().numpy()
                self.grid_result.append(result)
        self.grid_result = np.array(self.grid_result, dtype=np.int).transpose()

    def calc_reward(self):
        self.ui.amg_thread.reward = 0
        reward_score = [i + 1 for i in range(len(self.ui.ref_images))]
        for i in range(len(self.ui.ref_images)):
            self.ui.amg_thread.reward += reward_score[i] * len(self.grid_result[np.where(self.grid_result == i)])
