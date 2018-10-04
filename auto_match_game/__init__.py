from gym.envs.registration import register
from auto_match_game.auto_match_game_env import AutoMatchGameEnv

register(
    id='auto-match-game-v0',
    entry_point='auto_match_game:AutoMatchGameEnv'
)
