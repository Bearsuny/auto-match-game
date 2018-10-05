from gym.envs.registration import register

register(
    id='auto-match-game-v0',
    entry_point='auto_match_game.envs:AutoMatchGameEnv'
)
