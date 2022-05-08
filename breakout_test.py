# https://www.youtube.com/watch?v=hCeJeq8U0lo


import random
from neural_network import NetworkState, NeuralNetwork
#from ale_py import ALEInterface
import gym




def main():
    #ale = ALEInterface()
    #env = gym.make("Breakout-v0")
    env = gym.make("Breakout-v0", render_mode="human")
    #env = gym.make("Breakout-v0", render_mode="rgb_array")
    #env = gym.wrappers.ResizeObservation(env, (84, 84))

    height, width, channels = env.observation_space.shape
    #actions = env.observation_space.n

    observ_space = env.observation_space
    action_space = env.action_space

    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    #print("actions:", actions)

    print("observ_space:", observ_space)
    print("action_space:", action_space)

    print("actions:", env.unwrapped.get_action_meanings())

    episodes = 5
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            #env.render()
            #env.render(mode="rgb_array")
            action = random.choice([0,1,2,3])
            n_state, reward, done, info = env.step(action)
            score += reward
        print("Episode: {}, Score: {}".format(episode, score))
    env.close()

























if __name__ == "__main__":
    main()
