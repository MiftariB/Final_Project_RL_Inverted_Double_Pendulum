import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import numpy as np
import pybullet_envs
import time

env = gym.make("InvertedDoublePendulumBulletEnv-v0")
env.render(mode="human")
obs = env.reset()
while 1:
	time.sleep(1/60)
	env.step([10])
