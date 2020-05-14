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
    xtrain = np.zeros((len(tuples),10))
    ytrain = np.zeros(len(tuples))
    j = 0
    #generate intial training set
    for prev_state,action,reward in tuples:

        for i in range(9):
            xtrain[j][i] = prev_state[i]


        xtrain[j][9] = action[0]

        ytrain[j] = reward

        j=j+1

    return xtrain, ytrain

def BuildQDataset(tuples, Qprev, actions, gamma):
    # building dataset
    xtrain = np.zeros((len(tuples),10))
    ytrain = np.zeros(len(tuples))

    j = 0
    #generate intial training set
    for i in tqdm(range(len(tuples))):

        prev_state,action,reward = tuples[i]

        for i in range(9):
            xtrain[j][i] = prev_state[i]

        xtrain[j][9] = action[0]

        max_rew, max_action = Max(Qprev, actions, xtrain[j])
        xtrain[j][9] = max_action

        ytrain[j] = ytrain[j] + gamma*max_rew

        j=j+1


    return xtrain, ytrain


def Max(Qprev, actions, xtrain):

    best_rew =  -100000
    best_action  = -100000

    for action in actions:

        xtrain[9] = action
        pred =  Qprev.predict([xtrain])

        if pred > best_rew:
            best_rew = pred
            best_action = action


    return best_rew, best_action



def GenerateActionSet(steps_actions):
    return np.arange(-1, 1, steps_actions).tolist()


def BuildFQI(N, tuples, gamma, N_trees, steps_actions):
    xtrain, ytrain =  BuildDataset(tuples)
    actions =  GenerateActionSet(steps_actions)
    Qprev  = ExtraTreesRegressor(n_estimators=N_trees)
    Qprev = Qprev.fit(xtrain, ytrain)
    return FQI(N, tuples, Qprev, actions, gamma)

def FQI(N, tuples, Qprev, actions, gamma):


    #Fitted Q algorithm
    for i in range(N):
        print("\n ITERATION " + str(i) + "/" + str(N-1))
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

  N = 50
  gamma = 0.95
  N_trees = 100
  steps_actions = 0.01
  actions =  GenerateActionSet(steps_actions)
  name = "FQImodel100_" + str(N) + "_" + str(gamma) + "_" + str(N_trees) + ".pickle"


  with open("FQImodel100_10_0.8.pickle", 'rb') as handle:
    fqi = pickle.load(handle)

  #fqi =  BuildFQI(N, tuples, gamma, N_trees, steps_actions)
  # # save the model
  # with open(name, 'wb') as handle:
  #     pickle.dump(fqi, handle, protocol=pickle.HIGHEST_PROTOCOL)

  sum_r = 0

  for z in range(100):
    frame = 0
    score = 0
    restart_delay = 0
    obs = env.reset()
    r = 0
    x = np.zeros((1,7))

    while 1:
      time.sleep(1. / 60.)

      for i in range(9):
          x[0][i] = obs[i]

      x[0][9] = 0
      rew, action = Max(fqi, actions, x[0])

      a = [action]
      #a = [0]
      obs, r, done, _ = env.step(a)
      score += r
      frame += 1
      still_open = env.render("human")
      print("\n test")
      print(still_open)
      if still_open == False:
        return
      print("test2\n")
      if not done: continue
      if restart_delay == 0:
        print("score=%0.2f in %i frames" % (score, frame))
        restart_delay = 60 * 2  # 2 sec at 60 fps
      else:
        restart_delay -= 1
        if restart_delay > 0: continue
        break

    sum_r += r

  sum_r/=100
  print( "\n AVERAGE REWARD OF A GAME IS " + str(sum_r))

if __name__ == "__main__":
  main()
