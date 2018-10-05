import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPalette
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal

from auto_match_game.ui.config import AutoMatchGameUIConfig as UIC
from reward_grid.config import RewardGridConfig as RGC
from reward_grid.net_eval import get_reward_grid_roi

import torch


class AutoMatchGameUI(QWidget):
    def __init__(self):
        super().__init__()
        self.ref_images = os.listdir(UIC.reward_ref_image_path)
        self.positions = [(i, j) for i in range(UIC.reward_row) for j in range(UIC.reward_col)]
        self.grid_result = []

        self.reward_grid_net = torch.load(RGC.reward_grid_net_save_path)
        self.reward_grid_net.to(self.reward_grid_net.device)
        self.reward_grid_net.eval()

        self.widget1 = QWidget(self)
        self.widget2 = QWidget(self)

        self.hbox_layout = QHBoxLayout()
        self.vbox_layout = QVBoxLayout()
        self.grid_layout = QGridLayout()

        self.step_label = QLabel()
        self.score_label = QLabel()
        self.train_button = QPushButton()

        self.label_list = []
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(*UIC.widget_size)
        self.setWindowTitle(UIC.title)

        self.widget1.setFixedSize(*UIC.widget1_size)
        self.widget2.setFixedSize(*UIC.widget2_size)

        self.hbox_layout.setContentsMargins(0, 0, 0, 0)
        self.hbox_layout.setSpacing(0)

        self.vbox_layout.setContentsMargins(0, 0, 0, 0)
        self.vbox_layout.setSpacing(0)

        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(0)

        for position in self.positions:
            pick_num = np.random.randint(0, len(self.ref_images))
            pixmap = QPixmap(os.path.join(UIC.reward_ref_image_path, self.ref_images[int(pick_num)]))
            pixmap = pixmap.scaled(*UIC.scale_size)

            pe = QPalette()
            if sum(position) % 2 == 0:
                pe.setColor(QPalette.Background, QColor(106, 100, 102))
            else:
                pe.setColor(QPalette.Background, QColor(76, 74, 75))

            lbl = QLabel(self)
            lbl.setAutoFillBackground(True)
            lbl.setPalette(pe)
            lbl.setPixmap(pixmap)
            lbl.setAlignment(Qt.AlignCenter)
            self.label_list.append(lbl)

            self.grid_layout.addWidget(lbl, *position)

        self.step_label.setText(f'STEP: 10')
        self.score_label.setText(f'SCORE: 100')
        self.train_button.setText(f'TRAIN')
        font = QFont()
        font.setPixelSize(36)
        self.setFont(font)

        self.train_button.setFixedWidth(300)
        pe = QPalette()
        pe.setColor(QPalette.Button, Qt.red)
        self.train_button.setPalette(pe)

        self.hbox_layout.addWidget(self.step_label, alignment=Qt.AlignCenter)
        self.hbox_layout.addWidget(self.score_label, alignment=Qt.AlignCenter)
        self.hbox_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)

        self.widget1.setLayout(self.grid_layout)
        self.widget2.setLayout(self.hbox_layout)
        self.vbox_layout.addWidget(self.widget1, alignment=Qt.AlignCenter)
        self.vbox_layout.addWidget(self.widget2, alignment=Qt.AlignCenter)
        self.setLayout(self.vbox_layout)

        self.move_center()

    def move_center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def generate_grid_result(self):
        self.grid_result = []
        pix = self.grab()
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
        print(self.grid_result)

    def reset(self):
        for position in self.positions:
            pick_num = np.random.randint(0, len(self.ref_images))
            pixmap = QPixmap(os.path.join(UIC.reward_ref_image_path, self.ref_images[int(pick_num)]))
            pixmap = pixmap.scaled(*UIC.scale_size)

            self.label_list[position[0] * UIC.reward_row + position[1]].setPixmap(pixmap)
            lbl = QLabel(self)
            lbl.setAlignment(Qt.AlignCenter)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    amgui = AutoMatchGameUI()
    amgui.show()
    sys.exit(app.exec_())
