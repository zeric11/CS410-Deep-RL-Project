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
                new_input.append((r + g + b) / 3)
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
            return value

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
    params.env_name = "Breakout-v0"
    params.neural_network = None
    params.input_size = 210 * 160 * 2
    params.hidden_amount = 3
    params.hidden_size = 100
    params.output_size = 4
    params.learning_rate = 0.1
    params.momentum_value = 0.9
    params.momentum_enabled = False
    params.alpha = 1
    params.gamma = 0.95
    params.epsilon = 100
    params.epsilon_decay = 5
    params.batch_size = 150
    params.episodes_amount = 100
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
    plt.title("Test 1")
    plt.savefig()
    plt.show()


def training(params: TrainingParams) -> Tuple[List[int], List[float]]:
    env = gym.make(params.env_name, render_mode="rgb_array")
    if not params.neural_network:
        params.neural_network = NeuralNetwork(params.input_size, params.hidden_amount, params.hidden_size, params.output_size, \
                                              params.learning_rate, params.momentum_value, params.momentum_enabled)

    viewer = rendering.SimpleImageViewer()

    height, width, channels = env.observation_space.shape
    observ_space = env.observation_space
    action_space = env.action_space
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("observ_space:", observ_space)
    print("action_space:", action_space)
    print("actions:", env.unwrapped.get_action_meanings())

    x_values = []
    y_values = []

    epsilon = params.epsilon
    for episode in range(params.episodes_amount):
        history = History()
        pop_amount = int(params.batch_size / 2)

        env_state = None
        if params.filters_enabled:
            env_state = params.conv_layer.generate_filtered_input(env.reset())
        else:
            env_state = params.conv_layer.generate_unfiltered_input(env.reset())

        prev_env_state = None
        score = 0
        prev_score = 0
        prev_lives = 5

        done = False
        step_number = 1
        afk_counter = 1
        afk_check_size = 100
        while not done:
            rgb_values = env.render("rgb_array")
            viewer.imshow(np.repeat(np.repeat(rgb_values, 3, axis=0), 3, axis=1))

            network_state = None
            action = 1
            if step_number > 1:
                network_state = params.neural_network.execute_forward_propagation(prev_env_state + env_state)
                action = random.randrange(0, params.output_size) if random.randrange(0, 100) < epsilon else network_state.choose_action()

            observation, reward, done, info = env.step(action)
            score += reward

            lives = info["lives"]
            if lives < prev_lives:
                reward -= 1
                prev_lives = lives

            if afk_counter == afk_check_size and score <= prev_score:
                reward -= 5
                prev_score = score
                afk_counter = 0

            if step_number > 1:
                if params.display_outputs_enabled:
                    print("Ep:", episode + 1, "\tStep:", step_number, end='\t', flush=True)
                    network_state.display_output()
                    print("\tAction:", action, "\tReward:", reward)
                history.add_event(network_state, action, reward)

            if history.get_length() >= params.batch_size:
                # After certain number of steps has been completed, we are left with a "history" of network_states.
                # A batch update must be performed to update the neural network where the reward earned at the 
                # last action is passed down through the previous actions and the network's weights are adjusted 
                # according to these rewards.
                history.update_neural_network_pop_amount(params.neural_network, pop_amount, params.alpha, params.gamma)
            
            prev_env_state = env_state
            if params.filters_enabled:
                env_state = params.conv_layer.generate_filtered_input(observation)
            else:
                env_state = params.conv_layer.generate_unfiltered_input(observation)

            step_number += 1
            afk_counter += 1

        while history.get_length() > 0:
            history.update_neural_network_pop_amount(params.neural_network, pop_amount, params.alpha, params.gamma)

        epsilon -= params.epsilon_decay

        print("Episode: {}, Score: {}".format(episode + 1, score))
        x_values.append(episode + 1)
        y_values.append(score)

    env.close()


if __name__ == "__main__":
    main()



