
import matplotlib.pyplot as plt
import gym
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from ale_py import ALEInterface
from ale_py.roms import Breakout
ale = ALEInterface()
#ale.loadROM(MontezumaRevenge)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import reinforcement_learning as rl
env_name = 'ALE/MontezumaRevenge-v5'
#env_name = 'ALE/MsPacman-v5'

rl.checkpoint_base_dir = 'checkpoints_tutorial16/'
rl.update_paths(env_name=env_name)

agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=True)

model = agent.model
replay_memory = agent.replay_memory
agent.run(num_episodes=533)#1000)#30

log_q_values = rl.LogQValues()
log_reward = rl.LogReward()

#print(help(log_q_values))  #
#log_q_values.read()
#print(log_reward)
log_reward.read()

#Testing
print('Testing')
agent.epsilon_greedy.epsilon_testing
agent.training = False
agent.reset_episode_rewards()
agent.render = True
agent.run(num_episodes=30)#1000

#Mean reward
print('Mean reward')
agent.reset_episode_rewards()
agent.render=False
agent.run(num_episodes=30)

rewards = agent.episode_rewards
print("Rewards for {0} episodes:".format(len(rewards)))
print("- Min:   ", np.min(rewards))
print("- Mean:  ", np.mean(rewards))
print("- Max:   ", np.max(rewards))
print("- Stdev: ", np.std(rewards))
#epsilon testing epsilon 0.05
#epsilon start value: 1, end value:0.001
#number of iterations = 1e9
