# https://www.youtube.com/watch?v=hCeJeq8U0lo

import faulthandler
faulthandler.enable()

from typing import List
import random
from neural_network import NetworkState, NeuralNetwork
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


class Event:
    def __init__(self, network_state: NetworkState, chosen_action: int, reward: int) -> None:
        self.network_state = network_state
        self.chosen_action = chosen_action
        self.reward = reward

    def __del__(self):
        del self.network_state


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

    neural_network = NeuralNetwork(210 * 160, 1, 100, 4)

    alpha = 0.5
    gamma = 0.9
    epsilon = 100

    episodes = 100
    for episode in range(episodes):
        env_state = generate_unfiltered_input(env.reset())
        prev_env_state = None
        done = False
        score = 0

        history = []

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
                    action = np.argmax(network_state.output_layer)

            observation, reward, done, info = env.step(action)
            score += reward

            if first_step_done:
                history.insert(0, Event(network_state, action, reward))
            else:
                first_step_done = True

            if len(history) == 100:
                update_network(neural_network, history, alpha, gamma)
            
            prev_env_state = env_state
            env_state = generate_unfiltered_input(observation)

            if done:
                break

        print("Episode: {}, Score: {}".format(episode, score))

        update_network(neural_network, history, alpha, gamma)

        if episode % 5 == 0:
            epsilon -= 10

    env.close()


def generate_unfiltered_input(rgb_values) -> List[int]:
    new_input = []
    for i in range(210):
        for j in range(160):
            r = int(rgb_values[i][j][0])
            g = int(rgb_values[i][j][1])
            b = int(rgb_values[i][j][2])
            new_input.append((r + g + b) / 3)
    return new_input


# After a game has been completed, we are left with a "history" of network_states.
# A batch update must be performed to update the neural network where the reward
# earned at the final action is passed down through the previous actions and 
# the network's weights are adjusted according to these rewards.
def update_network(network: NeuralNetwork, history: List[Event], alpha: float, gamma: float) -> None:
    first_event_done = False
    previous_max_Qvalue = 0
    while(history):
        event = history[0]
        network_state, chosen_action, reward = event.network_state, event.chosen_action, event.reward
        target = network_state.output_layer.copy()
        if not first_event_done:
            target[chosen_action] += alpha * reward
            first_event_done = True
        else:
            target[chosen_action] += \
                alpha * (reward + (gamma * previous_max_Qvalue) - max(network_state.output_layer))
        previous_max_Qvalue = max(target)
        network.execute_back_progagation(network_state, target)
        
        history.pop(0)

























if __name__ == "__main__":
    main()



