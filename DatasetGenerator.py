#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import numpy as np
import pybullet_envs
import time
import random
import pickle


def relu(x):
  return np.maximum(x, 0)




def main():
    env = gym.make("InvertedDoublePendulumBulletEnv-v0")
    env.render(mode="human")

    tuples = []
    nb_of_games = 100
    for i in range(nb_of_games):
        print(i)
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        while 1:
            time.sleep(1. / 60.)
            a =  [random.uniform(-2.0, 2.0)]
            prev_obs =  obs
            # print("STEP: "+str(a))
            obs, r, done, _ = env.step(a)
            tuples.append( (prev_obs, a, obs, r) )
            score += r
            frame += 1
            still_open = env.render("human")
            if still_open == False:
                return
            if not done: continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60 * 2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
            if restart_delay > 0: continue
            break

    with open('100RandomDataset.pickle', 'wb') as handle:
        pickle.dump(tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
  main()
