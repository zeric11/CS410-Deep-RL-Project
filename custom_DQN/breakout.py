# https://www.youtube.com/watch?v=hCeJeq8U0lo
# valgrind --log-file="mem_test.txt" --tool=memcheck --leak-check=yes -s python3 breakout.py

import faulthandler
faulthandler.enable()

from typing import List, Tuple
import random
from DQN import ConvLayer, NetworkState, NeuralNetwork, History
#from conv_layer import ConvLayer
import numpy as np
#from ale_py import ALEInterface
import gym
from gym.envs.classic_control import rendering
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
    params.env_name = "Breakout-v0"
    params.initial_image_height = 210
    params.initial_image_width = 160
    params.final_image_height = 84
    params.final_image_width = 84
    params.step_skip_amount = 4
    params.neural_network = None
    params.hidden_amount = 2
    params.hidden_size = 1000
    params.output_size = 4
    params.learning_rate = 0.0001
    params.momentum_value = 0.1
    params.momentum_enabled = True
    params.randomize_weights = True
    params.alpha = 0.1
    params.gamma = 0.9
    params.epsilon = 100
    params.epsilon_decay = 0.01
    params.batch_size = 30
    params.episodes_amount = 1000
    params.display_outputs_enabled = True
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
    plt.plot(x_values, y_values)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Test 7")
    plt.savefig("Test 7")
    plt.show()


def training(params: TrainingParams) -> Tuple[List[int], List[float]]:
    x_values = []
    y_values = []

    env = gym.make(params.env_name, render_mode="rgb_array")
    if not params.neural_network:
        input_size = params.final_image_height * params.final_image_width * params.step_skip_amount
        params.neural_network = NeuralNetwork(input_size, params.hidden_amount, params.hidden_size, params.output_size, \
                                              params.learning_rate, params.momentum_value, params.momentum_enabled, params.randomize_weights)

    conv_layer = ConvLayer(params.initial_image_height, params.initial_image_width, params.final_image_height, \
                        params.final_image_width, params.step_skip_amount)
    for filter in params.filters:
        conv_layer.add_filter(filter)

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
        step_batch_size = 4
        afk_counter = 0
        afk_max_amount = 150
        afk_reward = -1000
        afk_reward_growth = -5
        #reward_coefficient = 5
        while not done:
            network_state = params.neural_network.execute_forward_propagation(conv_layer)
            conv_layer.clear_images()
            action = random.randrange(0, params.output_size) if random.randrange(0, 100) < epsilon else network_state.choose_action()

            step_batch_reward = 0
            for i in range(step_batch_size):
                rgb_values = env.render("rgb_array")
                viewer.imshow(np.repeat(np.repeat(rgb_values, 3, axis=0), 3, axis=1))

                observation, reward, done, info = env.step(action)
                score += reward

                conv_layer.add_image(observation)

                if score <= prev_score:
                    afk_counter += 1
                else:
                    afk_counter = 0
                prev_score = score

                lives = info["lives"]
                if lives < prev_lives:
                    #reward -= 5 * (5 - lives)
                    reward -= 1
                    prev_lives = lives
                    afk_counter = 0

                if afk_counter == afk_max_amount:
                    reward += afk_reward
                    #afk_reward += afk_reward_growth
                    afk_counter = 0

                if params.display_outputs_enabled:
                    #actual_reward = reward * reward_coefficient
                    print("Ep:", episode, "\tStep:", step_number, end='\t', flush=True)
                    network_state.display_output()
                    print("\tAction:", action, "\tReward:", reward)

                step_number += 1
                step_batch_reward += reward

            history.add_event(network_state, action, step_batch_reward)
            if history.get_length() >= params.batch_size:
            #if history.get_length() >= episode:
                # After certain number of steps has been completed, we are left with a "history" of network_states.
                # A batch update must be performed to update the neural network where the reward earned at the 
                # last action is passed down through the previous actions and the network's weights are adjusted 
                # according to these rewards.
                history.update_neural_network_last_event(params.neural_network, params.alpha, params.gamma)

        history.update_neural_network_all_events(params.neural_network, params.alpha, params.gamma)

        epsilon -= epsilon * 0.01
        if epsilon < 0.01:
            epsilon = 0

        print("Episode: {}, Score: {}".format(episode, score))
        x_values.append(episode)
        y_values.append(score)

    env.close()

    return (x_values, y_values)


if __name__ == "__main__":
    main()



