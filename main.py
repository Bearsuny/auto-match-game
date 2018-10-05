import sys
from PyQt5.QtWidgets import QApplication
import gym
import auto_match_game

if __name__ == '__main__':
    app = QApplication(sys.argv)
    env = gym.make('auto-match-game-v0')
    sys.exit(app.exec_())
