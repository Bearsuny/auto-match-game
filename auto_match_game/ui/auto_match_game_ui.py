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

from auto_match_game.ui.config import AutoMatchGameUIConfig as UIC
from reward_grid.config import RewardGridConfig as RGC


class AutoMatchGameUI(QWidget):
    def __init__(self):
        super().__init__()
        self.ref_images = []

        for reward_item in RGC.reward_name:
            for ref_item in os.listdir(UIC.reward_ref_image_path):
                if reward_item in ref_item:
                    self.ref_images.append(ref_item)

        self.positions = [(i, j) for i in range(UIC.reward_row) for j in range(UIC.reward_col)]

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

        self.step_label.setText(f'STEP: None')
        self.score_label.setText(f'SCORE: None')
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    amgui = AutoMatchGameUI()
    amgui.show()
    sys.exit(app.exec_())
