import gym


env = gym.make('LunarLander-v2')

env.reset()

print(env.action_space.sample())

print(env.observation_space.shape)
print(env.observation_space.sample())


env.close()
