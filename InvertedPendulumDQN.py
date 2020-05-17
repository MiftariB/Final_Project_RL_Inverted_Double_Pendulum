#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import math
import numpy as np
import random
import pybullet_envs
import time
import pickle
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def GenerateActionSet(steps_actions):
    return np.arange(-0.5, 0.5, steps_actions).tolist()

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 20
N = 50
STEP_ACTIONS = 0.05
N_INPUTS =  9 # variables of a state
N_GAMES = 10000
GAME_AV  = 20
FREQ_IM = 100

actions =  GenerateActionSet(STEP_ACTIONS)
n_actions = len(actions)
steps_done = 0

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# replay memory stores the encountered transitions
# a transition is a (prev_state, action, reward, next_state) tuple
# if the max capacity has been reached, and that another transition is pushed,
# the oldest transitions will be forgotten
class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# remembers a given amount of scores from nb_of_games
# can return the average score of all games scores in the memory
# when full, the next scores erases the oldest score
class ScoreMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, score):
        """Saves a score."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = score
        self.position = (self.position + 1) % self.capacity

    def average_score(self):
        return sum(self.memory) / len(self.memory)

# our deep Q learning network
# 'inputs' is the amount of elements used as input
# 'outputs' is the amount of elements used as outputs
class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.head = nn.Linear(1024, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.head(x.view(x.size(0), -1))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch  This converts batch-array of Transitions
    # to Transition of batch-arrays.

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.from_numpy(s) for s in batch.next_state
                                                if s is not None])

    # creating batches of states, actions and rewards
    # TODO : étrange : batch.reward n'a pas la même représentation que batch.action, ce qui me force a faire tout
    # un tas de manip : pourquoi ? Aussi, au final ils sont très légèrement différents, à surveiller
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(tuple([torch.from_numpy(np.array(a)).unsqueeze(0) for a in batch.action if a is not None]))
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.float()).gather(1, action_batch.long().view(-1,1) )

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    non_final_next_states = non_final_next_states.view(-1,9)
    next_state_values[non_final_mask] = target_net(non_final_next_states.float()).max(1)[0].detach()


    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))


    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



env = gym.make("InvertedDoublePendulumBulletEnv-v0")
env.render(mode="human")

# policy and target networks
policy_net = DQN(N_INPUTS, n_actions).to(device)
target_net = DQN(N_INPUTS, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.eval()

# get an optimizer for selecting the best action
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

#scores measures
score_mem = ScoreMemory(GAME_AV)
scores = []

#playing games
for z in range(N_GAMES):

    print("\n ITERATION " + str(z))

    frame = 0
    score = 0
    restart_delay = 0
    obs = env.reset()
    r = 0
    x = np.zeros((1,7))

    if z%FREQ_IM == 0 and z != 0:
        plt.plot(scores)
        plt.ylabel('average score on the 20 last games')
        plt.xlabel('nb of games played')
        plt.show()

        avg_score = score_mem.average_score()
        model_name = str(z) + 'eps_' + str(avg_score) + 'score.pt'
        torch.save(policy_net.state_dict(), model_name)

    #playing one game
    while 1:
        time.sleep(1. / 60.)

        # cast to tensor of appropriate dimension
        obs = torch.from_numpy(obs)
        obs = obs.unsqueeze(0)

        #select best action with epsilon greedy policy
        action_i = select_action(obs.float())

        #mapping the action chosen by the network by the corresponding action
        action = actions[action_i]

        # applying the action
        prev_obs = obs
        obs, r, done, _ = env.step([action])

        #TEST : TODO
        if done:
            r = torch.tensor([0.0], device=device)
        else:
            r = torch.tensor([r], device=device)

        # Store the transition in memory (the action needs to be the indexed action here)
        memory.push((prev_obs, action_i, obs, r))

        # Perform one step of the optimization (on the target network)
        optimize_model()

        #update elements
        score += r
        frame += 1
        still_open = env.render("human")

        #closing conditions
        if still_open == False:
            print('\n WARNING !!!! THIS USED TO BE A RETURN, NOT SURE THE BREAK AS APPROPRIATE \n')
            break

        if not done: continue

        if restart_delay == 0:
            print("score=%0.2f in %i frames" % (score, frame))
            # update scores measures
            score_mem.push(float(score))
            restart_delay = 6  # 2 sec at 60 fps
        else:
            restart_delay -= 1
        if restart_delay > 0: continue
        break

    # Update the target network, copying all weights and biases in DQN
    if z % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


    print("Average score on the last 20 games : " + str(score_mem.average_score()))
    scores.append(score_mem.average_score())

plt.plot(scores)
plt.ylabel('average score on the 20 last games')
plt.xlabel('nb of games played')
plt.show()
