import gym
from stable_baselines3 import PPO
from pathlib import Path

# create a folder to save the model
models_path = Path(__file__).parent / 'models'
PPO_model_path = models_path / 'PPO'
A2C_model_path = models_path / 'A2C'
# create the folders if it does not exist
PPO_model_path.mkdir(parents=True, exist_ok=True)
A2C_model_path.mkdir(parents=True, exist_ok=True)

# log the training
log_path = Path(__file__).parent / 'logs'
# make the folder if it does not exist
log_path.mkdir(parents=True, exist_ok=True)


env = gym.make('LunarLander-v2')
env.reset()

model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=str(log_path))

TIMESTEPS = 20000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name=f'PPO_{i}')
    model.save(str(PPO_model_path / f'PPO_{TIMESTEPS * i}'))

# run for 1000 steps which is about 10 seconds
'''N_EPISODES = 10
for episode in range(N_EPISODES):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs , reward , done , info = env.step(env.action_space.sample())
'''
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
