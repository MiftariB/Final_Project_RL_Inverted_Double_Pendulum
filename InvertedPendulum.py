#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import numpy as np
import pybullet_envs
import time
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm


def BuildDataset(tuples):
    # building dataset
    xtrain = np.zeros((len(tuples),14))
    ytrain = np.zeros(len(tuples))
    j = 0
    #generate intial training set
    for prev_state,action,next_state,reward in tuples:

        for i in range(6):
            xtrain[j][i] = prev_state[i]


        xtrain[j][6] = action[0]

        for i in range(6):
            xtrain[j][i + 7] = next_state[i]

        ytrain[j] = reward

        j=j+1

    return xtrain, ytrain

def BuildQDataset(tuples, Qprev, actions, gamma):
    # building dataset
    xtrain = np.zeros((len(tuples),14))
    ytrain = np.zeros(len(tuples))

    j = 0
    #generate intial training set
    for i in tqdm(range(len(tuples))):

        prev_state,action,next_state,reward = tuples[i]

        for i in range(6):
            xtrain[j][i] = prev_state[i]

        xtrain[j][6] = action[0]

        for i in range(6):
            xtrain[j][i + 7] = next_state[i]

        max_rew, max_action = Max(Qprev, actions, xtrain[j])
        xtrain[j][6] = max_action

        ytrain[j] = reward + gamma*max_rew

        j=j+1


    return xtrain, ytrain


def Max(Qprev, actions, xtrain):

    best_rew =  -100000
    best_action  = -100000

    for action in actions:

        xtrain[6] = action
        pred =  Qprev.predict([xtrain])

        if pred > best_rew:
            best_rew = pred
            best_action = action


    return best_rew, best_action



def GenerateActionSet():
    return np.arange(-2, 2, 0.1).tolist()


def BuildFQI(N, tuples, gamma):
    xtrain, ytrain =  BuildDataset(tuples)
    actions =  GenerateActionSet()
    Qprev  = ExtraTreesRegressor(n_estimators=10)
    Qprev = Qprev.fit(xtrain, ytrain)
    return FQI(N, tuples, Qprev, actions, gamma)

def FQI(N, tuples, Qprev, actions, gamma):


    #Fitted Q algorithm
    for i in range(N):
        print("\n ITERATION " + str(i))
        j = 0

        print("building new dataset")
        xtrain, ytrain = BuildQDataset(tuples, Qprev, actions, gamma)

        Qcurr = Qprev
        print("fitting new model")
        Qcurr = Qcurr.fit(xtrain, ytrain)
        Qprev = Qcurr

    print("FQI done iterating")
    return Qcurr



def main():
  env = gym.make("InvertedDoublePendulumBulletEnv-v0")
  env.render(mode="human")

  with open('100RandomDataset.pickle', 'rb') as handle:
    tuples = pickle.load(handle)

  fqi =  BuildFQI(10, tuples, 0.1 )

  while 1:
    frame = 0
    score = 0
    restart_delay = 0
    obs = env.reset()
    r = 0

    while 1:
      time.sleep(1. / 60.)
      a = [-0.1]
      print("STEP: "+str(a) + " OBS: " + str(obs) + " r: " + str(r) + "\n")
      obs, r, done, _ = env.step(a)
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

if __name__ == "__main__":
  main()
