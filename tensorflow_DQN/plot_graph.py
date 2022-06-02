import numpy as np
import reinforcement_learning as rl
import matplotlib.pyplot as plt


rl.checkpoint_base_dir = 'checkpoints_tutorial16/'

#rl.update_paths(env_name = 'ALE/MontezumaRevenge-v5')
rl.update_paths(env_name = 'ALE/MsPacman-v5')

log_reward = rl.LogReward()
log_q_values = rl.LogQValues()
log_q_values.read()
log_reward.read()

#plt.plot(x,y,label)
number_of_states = np.arange(0,len(log_reward.count_states))
plt.plot(number_of_states, log_reward.episode, label='Episode Reward')
plt.plot(number_of_states, log_reward.mean, label='Mean of 100 episodes')
plt.xlabel('Episode-Count for Game Environment')

plt.legend()
plt.show()

#log_rewrd.txt
# numberofepisodesper_training?# num_of_state_per_episode? reward_per_episode? avg_reward_per_30_episodes?

#logqvalues
#num_of_episodes,num_states_processed, qval(min, mean, max, std)

num_of_states = np.arange(0,len(log_q_values.mean))
plt.plot(num_of_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()
