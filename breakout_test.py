# https://www.youtube.com/watch?v=hCeJeq8U0lo

import faulthandler
faulthandler.enable()

from typing import List
import random
from neural_network import NetworkState, NeuralNetwork, History
#from conv_layer import ConvLayer
import numpy as np
#from ale_py import ALEInterface
import gym


from gym.envs.classic_control import rendering


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)



def main():
    #ale = ALEInterface()
    #env = gym.make("Breakout-v0")
    #env = gym.make("Breakout-v0", render_mode="human")
    env = gym.make("Breakout-v0", render_mode="rgb_array")
    #env = gym.wrappers.ResizeObservation(env, (84, 84))

    height, width, channels = env.observation_space.shape
    observ_space = env.observation_space
    action_space = env.action_space

    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("observ_space:", observ_space)
    print("action_space:", action_space)
    print("actions:", env.unwrapped.get_action_meanings())

    viewer = rendering.SimpleImageViewer()

    neural_network = NeuralNetwork((210 - 2) * (160 - 2) * 2, 1, 50, 4)
    conv_layer = ConvLayer()
    conv_layer.add_filter([[ 0, 0, 0],
                           [ 1, 1, 1],
                           [-1,-1,-1]])

    alpha = 1
    gamma = 0.9
    epsilon = 100

    batch_size = 30

    episodes = 1000
    for episode in range(episodes):
        env_state = conv_layer.generate_unfiltered_input(env.reset())
        #env_state = conv_layer.generate_filtered_input(env.reset())
        prev_env_state = None
        done = False
        score = 0

        history = History()

        first_step_done = False
        while not done:
            rgb_values = env.render("rgb_array")
            viewer.imshow(repeat_upsample(rgb_values, 3, 3))

            network_state = None
            action = 1
            if first_step_done:
                network_state = neural_network.execute_forward_propagation(prev_env_state + env_state)
                if random.randrange(0, 100) < epsilon:
                    action = random.randrange(0, 4)
                else:
                    output = network_state.get_output()
                    print(output)
                    action = np.argmax(output)

            observation, reward, done, info = env.step(action)
            score += reward

            if first_step_done:
                history.add_event(network_state, action, reward)
            else:
                first_step_done = True

            if history.get_length() >= batch_size:
                # After certain number of steps has been completed, we are left with a "history" of network_states.
                # A batch update must be performed to update the neural network where the reward earned at the 
                # last action is passed down through the previous actions and the network's weights are adjusted 
                # according to these rewards.
                history.update_neural_network(neural_network, alpha, gamma)
            
            prev_env_state = env_state
            env_state = conv_layer.generate_unfiltered_input(observation)
            #env_state = conv_layer.generate_filtered_input(observation)

            if done:
                break

        print("Episode: {}, Score: {}".format(episode, score))

        while history.get_length() > 0:
            history.update_neural_network(neural_network, alpha, gamma)

        epsilon -= 10

    env.close()





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

        new_input = []
        for filter in self.filters:
            for i in range(len(rgb_values) - (len(filter) - 1)):
                for j in range(len(rgb_values[i]) - (len(filter[0]) - 1)):
                    new_input.append(apply_filter(filter, i, j))
        return new_input














if __name__ == "__main__":
    main()



