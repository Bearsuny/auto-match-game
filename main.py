import gym
import auto_match_game

if __name__ == '__main__':
    env = gym.make('auto-match-game-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        # env.step(env.action_space.sample())
