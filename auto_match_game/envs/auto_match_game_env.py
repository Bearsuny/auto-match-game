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
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


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
                QThread.msleep(EC.sleep_msec)
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

        self.action_space = spaces.MultiDiscrete(np.array([EC.grid_row, EC.grid_col]))

        self.grid_result = []

    def train(self):
        self.ui.amg_thread = AutoMatchGameThread(self)

        self.ui.amg_thread.step_signal.connect(self.step)
        self.ui.amg_thread.reset_signal.connect(self.reset)
        self.ui.amg_thread.render_signal.connect(self.render)

        self.ui.amg_thread.start()

    def step(self, action):
        self.generate_grid_result()
        x, y = action
        direction = self.generate_direction(action)
        flag1 = False
        pe = QPalette()
        pe.setColor(QPalette.Background, Qt.red)
        if direction == 0:
            if x - 1 >= 0:
                flag1 = True
                self.grid_result[x - 1][y], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x - 1][y]
                self.ui.label_list[x * UIC.reward_row + y].setPalette(pe)
                self.ui.label_list[(x - 1) * UIC.reward_row + y].setPalette(pe)
        if direction == 1:
            if y + 1 <= EC.grid_col - 1:
                flag1 = True
                self.grid_result[x][y + 1], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x][y + 1]
                self.ui.label_list[x * UIC.reward_row + y].setPalette(pe)
                self.ui.label_list[x * UIC.reward_row + (y + 1)].setPalette(pe)
        if direction == 2:
            if x + 1 <= EC.grid_row - 1:
                flag1 = True
                self.grid_result[x + 1][y], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x + 1][y]
                self.ui.label_list[x * UIC.reward_row + y].setPalette(pe)
                self.ui.label_list[(x + 1) * UIC.reward_row + y].setPalette(pe)
        if direction == 3:
            if y - 1 >= 0:
                flag1 = True
                self.grid_result[x][y - 1], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x][y - 1]
                self.ui.label_list[x * UIC.reward_row + y].setPalette(pe)
                self.ui.label_list[x * UIC.reward_row + y - 1].setPalette(pe)

        if flag1:
            grid_test = self.grid_result.copy()
            grid_old_result = self.grid_result.copy()
            self.grid_result = self.get_match_result(grid_test)
            if not ((self.grid_result == grid_old_result).all()):
                self.game_steps -= 1
                self.calc_reward()
            else:
                if direction == 0:
                    if x - 1 >= 0:
                        self.grid_result[x - 1][y], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x - 1][y]
                if direction == 1:
                    if y + 1 <= EC.grid_col - 1:
                        self.grid_result[x][y + 1], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x][y + 1]
                if direction == 2:
                    if x + 1 <= EC.grid_row - 1:
                        self.grid_result[x + 1][y], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x + 1][y]
                if direction == 3:
                    if y - 1 >= 0:
                        self.grid_result[x][y - 1], self.grid_result[x][y] = self.grid_result[x][y], self.grid_result[x][y - 1]
        else:
            print(action, direction)

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
        grid_test = self.grid_result.copy()
        self.grid_result = self.get_match_result(grid_test)

    def render(self, mode='human'):
        print(f'grid_result: {self.grid_result}')
        self.ui.step_label.setText(f'STEP: {self.game_steps}')
        self.ui.score_label.setText(f'SCORE: {self.ui.amg_thread.reward}')
        for position in self.ui.positions:
            pe = QPalette()
            if sum(position) % 2 == 0:
                pe.setColor(QPalette.Background, QColor(106, 100, 102))
            else:
                pe.setColor(QPalette.Background, QColor(76, 74, 75))
            pixmap = QPixmap(os.path.join(UIC.reward_ref_image_path, self.ui.ref_images[int(self.grid_result[position[0]][position[1]])]))
            pixmap = pixmap.scaled(*UIC.scale_size)
            self.ui.label_list[position[0] * UIC.reward_row + position[1]].setPixmap(pixmap)
            self.ui.label_list[position[0] * UIC.reward_row + position[1]].setPalette(pe)

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

    def generate_direction(self, action):
        x, y = action
        if x == 0:
            if y == 0:
                direction = np.random.randint(1, 3)
            elif y == EC.grid_col - 1:
                direction = np.random.randint(2, 4)
            else:
                direction = np.random.randint(1, 4)
        elif x == EC.grid_row - 1:
            if y == 0:
                direction = np.random.randint(0, 2)
            elif y == EC.grid_col - 1:
                direction = np.random.randint(0, 2)
                if direction == 1:
                    direction = 3
            else:
                direction = np.random.randint(0, 3)
                if direction == 2:
                    direction = 3
        else:
            if y == 0:
                direction = np.random.randint(0, 3)
            elif y == EC.grid_col - 1:
                direction = np.random.randint(1, 4)
                if direction == 1:
                    direction = 0
            else:
                direction = np.random.randint(0, 4)
        return direction

    def get_match_result(self, grid):
        match_flag = -1
        result_row = self.match(grid)
        result_col = self.match(grid, reverse=True)
        result_temp = []
        if result_row == [] and result_col != []:
            for item_col in result_col:
                result_temp.append(item_col)
        elif result_col == [] and result_row != []:
            for item_row in result_row:
                result_temp.append(item_row)
        elif result_col != [] and result_row != []:
            for item_row in result_row:
                for item_col in result_col:
                    if item_row & item_col != set():
                        result_temp.append(item_row | item_col)
                    else:
                        result_temp.append(item_row)
                        result_temp.append(item_col)
        result = []
        print(result_temp)
        print(len(result_temp))
        for i in range(len(result_temp)):
            flag = False
            for j in range(len(result_temp)):
                if i != j:
                    if result_temp[i].issubset(result_temp[j]):
                        flag = True
            if not flag:
                result.append(result_temp[i])

        print(f'result: {result}')
        if result != []:
            for i, item in enumerate(result):
                x, y = max(item)
                result[i].remove((x, y))
                grid[x][y] += 1
            for group in result:
                for item in group:
                    grid[item[0]][item[1]] = match_flag

        grid_temp = []
        for i in range(grid.shape[1]):
            col = grid[:, i]
            col_prefix = [match_flag] * len(col[np.where(col == match_flag)])
            col = [item for item in col if item != match_flag]
            col_prefix.extend(col)
            col = col_prefix
            grid_temp.append(col)
        grid_temp = np.array(grid_temp, dtype=np.int)
        grid_temp = grid_temp.transpose()
        x, y = np.where(grid_temp == match_flag)
        for x_item, y_item in zip(x, y):
            grid_temp[x_item][y_item] = np.random.randint(0, 8)

        if result == []:
            return grid
        else:
            return self.get_match_result(grid_temp)

    def match(self, grid, reverse=False):
        result = []
        if reverse:
            grid_copy = grid.transpose().copy()
        else:
            grid_copy = grid.copy()
        for i in range(EC.grid_row):
            c_item = None
            p_item = None
            n = 1
            for j in range(EC.grid_col):
                c_item = grid_copy[i][j]
                if j != 0:
                    p_item = grid_copy[i][j - 1]
                if p_item == c_item:
                    n += 1
                    if j == EC.grid_col - 1:
                        if n >= 3:
                            k_list = []
                            if p_item < EC.reward_category - 1:
                                if reverse:
                                    for k in range(1, n + 1):
                                        k_list.append((j - k, i))
                                else:
                                    for k in range(1, n + 1):
                                        k_list.append((i, j - k))
                                result.append(set(k_list))
                        n = 1
                else:
                    if n >= 3:
                        k_list = []
                        if p_item < EC.reward_category - 1:
                            if reverse:
                                for k in range(1, n + 1):
                                    k_list.append((j - k, i))
                            else:
                                for k in range(1, n + 1):
                                    k_list.append((i, j - k))
                            result.append(set(k_list))
                    n = 1
        return result
