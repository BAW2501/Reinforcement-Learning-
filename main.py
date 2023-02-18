import gym
from stable_baselines3 import PPO


env = gym.make('LunarLander-v2')
env.reset()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
# run for 1000 steps which is about 10 seconds
N_EPISODES = 10
for episode in range(N_EPISODES):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs , reward , done , info = env.step(env.action_space.sample())

env.close()


# download https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz with a command
#!wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
# extract to /home/codespace/.mujoco/mjpro150
# pip install mujoco-py
# pip install gym[all]
# pip install stable-baselines3
# pip install pyglet==1.5.27
# to run use 
#  xvfb-run -s "-screen 0 1400x900x24" python main.py
# u can install xvfb-run with sudo apt-get install xvfb
