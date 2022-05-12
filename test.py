# https://www.youtube.com/watch?v=hCeJeq8U0lo

import faulthandler
faulthandler.enable()

from typing import List
import random
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



    viewer = rendering.SimpleImageViewer()


    episodes = 5
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0

        counter = 0
        while not done:
            #env.render()
            rgb_values = env.render("rgb_array")
            viewer.imshow(repeat_upsample(rgb_values, 3, 3))

            #print("start")
            #print(list(rgb).pop())
            #print("stop")
            #return

            action = random.randrange(0, 4)
            n_state, reward, done, info = env.step(action)
            score += reward

            counter += 1


        print(counter)



        print("Episode: {}, Score: {}".format(episode, score))




    env.close()





















if __name__ == "__main__":
    main()



