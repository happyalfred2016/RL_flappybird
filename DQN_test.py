import torch.nn as nn
import numpy as np
import torch
from collections import deque
import random


# Define some Hyper Parameters
BATCH_SIZE = 32     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
TARGET_NETWORK_REPLACE_FREQ = 100       # How frequently target network updates
MEMORY_CAPACITY = 2000                  # The capacity of experience replay buffer
IMG_SHAPE = [3, 256, 144]
N_ACTIONS = 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(18432, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DQN(object):
    def __init__(self):
        # -----------Define 2 networks (target and training)------#
        self.eval_net, self.target_net = Net(), Net()
        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer
        self.memory = list([np.zeros([MEMORY_CAPACITY] + IMG_SHAPE),
                            np.zeros([MEMORY_CAPACITY]),
                            np.zeros([MEMORY_CAPACITY]),
                            np.zeros([MEMORY_CAPACITY] + IMG_SHAPE)])
        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[0][index, :] = s
        self.memory[1][index, :] = a
        self.memory[2][index, :] = r
        self.memory[3][index, :] = s_
        self.memory_counter += 1

    def choose_action(self, x):
        # add 1 dimension to input state x
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net(x)
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the
            # action of cart.
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def learn(self):
        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_s = torch.tensor(self.memory[0][index, :])
        b_a = torch.tensor(self.memory[1][index, :])
        b_r = torch.tensor(self.memory[2][index, :])
        b_s_ = torch.tensor(self.memory[3][index, :])
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()


dqn = DQN()


class Comm:
    def get_obs(self):
        # state, reward, done
        return 1, 1, 1

    def send_act(self):
        return 0


class FlappyBird:
    def __init__(self, comm):
        self.comm = comm

    def reset(self):
        while self.comm.get_obs()[2]:
            continue
        return self.comm.get_obs()[0]

    def step(self, a):
        self.comm.send_act(a)
        return self.comm.get_obs()


env = FlappyBird(Comm)

# Start training
print("\nCollecting experience...")
for i_episode in range(400):
    state = env.reset()
    ep_r = 0
    while True:
        # take action based on the current state
        action = dqn.choose_action(state)
        # obtain the reward and next state and some other information
        state_next, reward, done = env.step(action)

        # store the transitions of states
        dqn.store_transition(state, action, reward, state_next)
        ep_r += reward
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
        if done:
            break
        # use next state to update the current state.
        state = state_next
