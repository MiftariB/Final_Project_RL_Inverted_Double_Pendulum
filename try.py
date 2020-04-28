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


### GET STATE : 
print('Initial state '+str(env.robot.calc_state()))
#return np.array([
#            x, vx,
#            self.pos_x,
#            np.cos(theta), np.sin(theta), theta_dot,
#            np.cos(gamma), np.sin(gamma), gamma_dot,
#        ])

time.sleep(1/60)

### step takes action as input 
#returns state 
#        sum_of_rewards
#        Done -> if robot's y position +0.3 <=1
#        empty {} -> don't know its utility
state,reward,done,other =  env.step([0.01])
print('After move state '+ str(state))
print('reward '+ str(reward))
print('Done '+str(done))
print('IDK '+str(other))

### Get ROBOT x and y position
x_pos = env.robot.pos_x
y_pos = env.robot.pos_y

print("x pos "+str(x_pos))#X pos shall be the position of the "slider"
print("y pos "+str(y_pos))#UNKNOWN but could be position of highest point

