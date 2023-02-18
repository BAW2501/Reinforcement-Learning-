import gym


env = gym.make('LunarLander-v2',render_mode='rgb_array')

env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()
