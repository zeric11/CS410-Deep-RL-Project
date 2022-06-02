# valgrind --log-file="mem_test.txt" --tool=memcheck --leak-check=yes -s python3 montezuma.py

import faulthandler
faulthandler.enable()

from typing import List, Tuple
import random
from DQN import ConvLayer, NetworkState, NeuralNetwork, History
import numpy as np
import gym
import matplotlib.pyplot as plt


class TrainingParams:
    def __init__(self) -> None:
        self.env_name: str = None
        self.initial_image_height: int = None
        self.initial_image_width: int = None
        self.final_image_height: int = None
        self.final_image_width: int = None
        self.step_skip_amount: int = None
        self.filters: List[List[List[int]]] = None
        self.filters_amount:int = None
        self.filters_enabled: bool = None
        self.neural_network: NeuralNetwork = None
        self.hidden_amount: int = None
        self.hidden_size: int = None
        self.output_size: int = None
        self.learning_rate: float = None
        self.momentum_value: float = None
        self.momentum_enabled: bool = None
        self.randomize_weights: bool = None
        self.alpha: float = None
        self.gamma: float = None
        self.epsilon: float = None
        self.epsilon_decay: float = None
        self.batch_size: int = None
        self.episodes_amount: int = None
        self.display_outputs_enabled: bool = None


def main():
    params = TrainingParams()
    params.env_name = "MontezumaRevenge-v0"
    params.initial_image_height = 210
    params.initial_image_width = 160
    params.final_image_height = 105
    params.final_image_width = 80
    params.step_skip_amount = 6
    params.neural_network = None
    params.hidden_amount = 3
    params.hidden_size = 1024
    params.output_size = 9
    params.learning_rate = 0.01
    params.momentum_value = 0.1
    params.momentum_enabled = True
    params.randomize_weights = True
    params.alpha = 1
    params.gamma = 0.95
    params.epsilon = 100
    params.batch_size = 300
    params.episodes_amount = 4000
    params.display_outputs_enabled = False
    params.filters_enabled = True
    params.filters_amount = 4
    params.filters = [[[-1,-1,-1], 
                       [ 1, 1, 1], # Top
                       [ 0, 0, 0]],
                      [[ 0, 0, 0], 
                       [ 1, 1, 1], # Bottom
                       [-1,-1,-1]],
                      [[-1, 1, 0], 
                       [-1, 1, 0], # Left
                       [-1, 1, 0]],
                      [[ 0, 1,-1], 
                       [ 0, 1,-1], # Right
                       [ 0, 1,-1]]]

    x_values, y_values = training(params)
    y_avg_values = get_avg_data(x_values, y_values)

    save_data_to_file("test_montezuma.txt", params, x_values, y_values)

    plt.plot(x_values, y_values)
    plt.plot(x_values, y_avg_values)
    plt.legend(["Episode Reward", "Mean of 100 episodes"])
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("test_montezuma")


def get_avg_data(x_values, y_values):
    def get_avg(start, end, array):
        counter = 0
        total = 0
        for i in range(start, end):
            total += array[i]
            counter += 1
        return total / counter

    y_avg_values = []
    for i in range(len(x_values)):
        start = i - 50 if i >= 50 else 0
        end = i + 50 + 1 if i + 50 < len(x_values) else len(x_values)
        avg = get_avg(start, end, y_values)
        y_avg_values.append(avg)
    return y_avg_values


def save_data_to_file(filename, params, x_values, y_values):
    with open(filename, "w") as data_file:
        data_file.write("env_name: " + str(params.env_name) + "\n")
        data_file.write("initial_image_height: " + str(params.initial_image_height) + "\n")
        data_file.write("initial_image_width: " + str(params.initial_image_width) + "\n")
        data_file.write("final_image_height: " + str(params.final_image_height) + "\n")
        data_file.write("final_image_width: " + str(params.final_image_width) + "\n")
        data_file.write("step_skip_amount: " + str(params.step_skip_amount) + "\n")
        data_file.write("hidden_amount: " + str(params.hidden_amount) + "\n")
        data_file.write("hidden_size: " + str(params.hidden_size) + "\n")
        data_file.write("output_size: " + str(params.output_size) + "\n")
        data_file.write("learning_rate: " + str(params.learning_rate) + "\n")
        data_file.write("momentum_value: " + str(params.momentum_value) + "\n")
        data_file.write("momentum_enabled: " + str(params.momentum_enabled) + "\n")
        data_file.write("randomize_weights: " + str(params.randomize_weights) + "\n")
        data_file.write("alpha: " + str(params.alpha) + "\n")
        data_file.write("gamma: " + str(params.gamma) + "\n")
        data_file.write("batch_size: " + str(params.batch_size) + "\n")
        data_file.write("episodes_amount: " + str(params.episodes_amount) + "\n")
        data_file.write("display_outputs_enabled: " + str(params.display_outputs_enabled) + "\n")
        data_file.write("filters_enabled: " + str(params.filters_enabled) + "\n")
        data_file.write("filters_amount: " + str(params.filters_amount) + "\n")
        data_file.write("\nEpisode\tScore\n")
        for i in range(len(x_values)):
            data_file.write(str(x_values[i]) + "\t" + str(y_values[i]) + "\n")


def training(params: TrainingParams) -> Tuple[List[int], List[float]]:
    x_values = []
    y_values = []

    env = gym.make(params.env_name, render_mode="rgb_array")
    height, width, channels = env.observation_space.shape
    observ_space = env.observation_space
    action_space = env.action_space
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("observ_space:", observ_space)
    print("action_space:", action_space)
    print("actions:", env.unwrapped.get_action_meanings())

    if not params.neural_network:
        input_size = params.final_image_height * params.final_image_width * 2
        if params.filters_enabled:
            input_size *= params.filters_amount
        params.neural_network = NeuralNetwork(input_size, params.hidden_amount, params.hidden_size, params.output_size, \
            params.learning_rate, params.momentum_value, params.momentum_enabled, params.randomize_weights)

    conv_layer = ConvLayer(params.initial_image_height, params.initial_image_width, params.final_image_height, \
        params.final_image_width, 2)
    if params.filters and params.filters_enabled:
        for filter in params.filters:
            conv_layer.add_filter(filter)

    epsilon = params.epsilon
    for episode in range(1, params.episodes_amount + 1):
        history = History()

        conv_layer.clear_images()
        conv_layer.add_image(env.reset())

        score = 0
        prev_score = 0
        prev_lives = 5

        done = False
        step_number = 1
        step_skip_amount = params.step_skip_amount
        afk_counter = 0
        afk_max_amount = 1000
        afk_reward = -1000
        while not done:
            network_state = params.neural_network.execute_forward_propagation(conv_layer)
            conv_layer.clear_images()
            action = random.randrange(0, params.output_size) if random.randrange(0, 100) < epsilon else network_state.choose_action()

            step_batch_reward = 0
            for i in range(step_skip_amount):
                observation, reward, done, info = env.step(action)
                score += reward

                if i == 0 or i == step_skip_amount - 1:
                    conv_layer.add_image(observation)

                if score <= prev_score:
                    afk_counter += 1
                else:
                    afk_counter = 0
                prev_score = score

                lives = info["lives"]
                if lives < prev_lives:
                    reward += -100
                    prev_lives = lives
                    afk_counter = 0

                if afk_counter == afk_max_amount:
                    reward += afk_reward
                    afk_counter = 0

                if params.display_outputs_enabled:
                    print("Ep:", episode, "\tStep:", step_number, end='\t', flush=True)
                    network_state.display_output()
                    print("\tAction:", action, "\tReward:", reward)

                step_number += 1
                step_batch_reward += reward

            history.add_event(network_state, action, step_batch_reward)
            if history.get_length() >= params.batch_size:
                # After certain number of steps has been completed, we are left with a "history" of network_states.
                # A batch update must be performed to update the neural network where the reward earned at the 
                # last action is passed down through the previous actions and the network's weights are adjusted 
                # according to these rewards.
                history.update_neural_network_last_event(params.neural_network, params.alpha, params.gamma)

        history.update_neural_network_all_events(params.neural_network, params.alpha, params.gamma)

        if episode > 1000:
            epsilon -= 1
            if episode < 2000 and epsilon <= 1:
                epsilon = params.epsilon
        

        print("Episode: {}, Score: {}".format(episode, score))
        x_values.append(episode)
        y_values.append(score)

    env.close()

    return (x_values, y_values)


if __name__ == "__main__":
    main()



