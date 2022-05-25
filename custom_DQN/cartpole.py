# https://www.youtube.com/watch?v=hCeJeq8U0lo

import faulthandler
faulthandler.enable()

from typing import List, Tuple
import random
from DQN import NetworkState, NeuralNetwork, History
#from conv_layer import ConvLayer
import numpy as np
#from ale_py import ALEInterface
import gym
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt


class ConvLayer:
    def __init__(self) -> None:
        self.filters = []

    def add_filter(self, filter) -> None:
        self.filters.append(filter)

    def generate_unfiltered_input(self, rgb_values) -> List[int]:
        new_input = []
        for i in range(210):
            for j in range(160):
                r = int(rgb_values[i][j][0])
                g = int(rgb_values[i][j][1])
                b = int(rgb_values[i][j][2])
                new_input.append((r + g + b) / (3 * 255))
        return new_input

    def generate_filtered_input(self, rgb_values) -> List[float]:
        def apply_filter(filter, i_index, j_index) -> float:
            value = 0
            for i in range(len(filter)):
                for j in range(len(filter[0])):
                    r = int(rgb_values[i + i_index][j + j_index][0])
                    g = int(rgb_values[i + i_index][j + j_index][1])
                    b = int(rgb_values[i + i_index][j + j_index][2])
                    value += filter[i][j] * ((r + g + b) / 3)
            return value / 255

        if(len(self.filters) == 0):
            return self.generate_filtered_input(rgb_values)

        new_input = []
        for filter in self.filters:
            for i in range(len(rgb_values) - (len(filter) - 1)):
                for j in range(len(rgb_values[i]) - (len(filter[0]) - 1)):
                    new_input.append(apply_filter(filter, i, j))
        return new_input


class TrainingParams:
    def __init__(self) -> None:
        self.env_name: str = None
        self.neural_network: NeuralNetwork = None
        self.conv_layer: ConvLayer = None
        self.filters_enabled: bool = None
        self.learning_rate: float = None
        self.momentum_value: float = None
        self.momentum_enabled: bool = None
        self.alpha: float = None
        self.gamma: float = None
        self.epsilon: float = None
        self.epsilon_decay: float = None
        self.batch_size: int = None
        self.episodes_amount: int = None
        self.display_outputs_enabled: bool = None


def main():
    params = TrainingParams()
    params.env_name = "CartPole-v1"
    params.neural_network = None
    params.input_size = 8
    params.hidden_amount = 5
    params.hidden_size = 100
    params.output_size = 2
    params.learning_rate = 0.000001
    params.momentum_value = 0.01
    params.momentum_enabled = False
    params.alpha = 0.001
    params.gamma = 0.9
    params.epsilon = 00
    params.epsilon_decay = 1
    params.batch_size = 00
    params.episodes_amount = 10000
    params.display_outputs_enabled = True
    params.conv_layer = ConvLayer()
    params.filters_enabled = False
    params.conv_layer.add_filter([[ 0, 0, 0],
                                  [ 1, 1, 1],
                                  [-1,-1,-1]])

    x_values, y_values = training(params)
    plt.plot(x_values, y_values)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Test 2")
    plt.savefig("Test 2")
    plt.show()


def training(params: TrainingParams) -> Tuple[List[int], List[float]]:
    x_values = []
    y_values = []

    env = gym.make(params.env_name)
    if not params.neural_network:
        params.neural_network = NeuralNetwork(params.input_size, params.hidden_amount, params.hidden_size, params.output_size, \
                                              params.learning_rate, params.momentum_value, params.momentum_enabled)

    viewer = rendering.SimpleImageViewer()

    #height, width, channels = env.observation_space.shape
    #observ_space = env.observation_space
    #action_space = env.action_space
    #print("height:", height)
    #print("width:", width)
    #print("channels:", channels)
    #print("observ_space:", observ_space)
    #print("action_space:", action_space)
    #print("actions:", env.unwrapped.get_action_meanings())

    epsilon = params.epsilon
    for episode in range(params.episodes_amount):
        history = History()

        env_state = env.reset()
        prev_env_state = env_state

        score = 0
        prev_score = 0
        prev_lives = 5

        done = False
        step_number = 1
        afk_counter = 0
        afk_max_amount = 100
        afk_reward = -100
        afk_reward_growth = -5
        #reward_coefficient = 5
        while not done:
            #rgb_values = env.render("rgb_array")
            #viewer.imshow(np.repeat(np.repeat(rgb_values, 3, axis=0), 3, axis=1))

            env.render()
        
            network_state = params.neural_network.execute_forward_propagation(np.concatenate((prev_env_state, env_state)))
            action = random.randrange(0, params.output_size) if random.randrange(0, 100) < epsilon else network_state.choose_action()

            #print(env.step(action))
            observation, reward, done, info = env.step(action)
            #print(env.step(action))
            score += reward

            if score <= prev_score:
                afk_counter += 1
            else:
                afk_counter = 0
            prev_score = score

            if afk_counter == afk_max_amount:
                reward += afk_reward
                #afk_reward += afk_reward_growth
                afk_counter = 0

            if done:
                reward = -100

            if params.display_outputs_enabled:
                #actual_reward = reward * reward_coefficient
                print("Ep:", episode + 1, "\tStep:", step_number, end='\t', flush=True)
                network_state.display_output()
                print("\tAction:", action, "\tReward:", reward)
            history.add_event(network_state, action, reward)

            if history.get_length() >= params.batch_size:
                # After certain number of steps has been completed, we are left with a "history" of network_states.
                # A batch update must be performed to update the neural network where the reward earned at the 
                # last action is passed down through the previous actions and the network's weights are adjusted 
                # according to these rewards.
                history.update_neural_network_last_event(params.neural_network, params.alpha, params.gamma)
            
            prev_env_state = env_state
            env_state = observation

            step_number += 1

        history.update_neural_network_all_events(params.neural_network, params.alpha, params.gamma)

        epsilon -= params.epsilon_decay

        print("Episode: {}, Score: {}".format(episode + 1, score))
        x_values.append(episode + 1)
        y_values.append(score)

    env.close()

    return (x_values, y_values)


if __name__ == "__main__":
    main()



