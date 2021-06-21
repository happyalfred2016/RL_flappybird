import torch.nn as nn
import numpy as np
import torch
import logging
import torchvision.models as models

# Define some Hyper Parameters
BATCH_SIZE = 16  # batch size of sampling process from buffer
LR = 0.0001  # learning rate
EPSILON = 0.9  # epsilon used for epsilon greedy approach
GAMMA = 0.99  # discount factor
TARGET_NETWORK_REPLACE_FREQ = 100  # How frequently target network updates
MEMORY_CAPACITY = 256  # The capacity of experience replay buffer
IMG_SHAPE = [3, 128, 72]
N_ACTIONS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.out(x)


def get_net():
    resnet18 = models.resnet18()
    num_ftrs = resnet18.fc.in_features
    resnet18.load_state_dict(torch.load('./resnet18-5c106cde.pth', map_location=device), strict=False)
    resnet18.fc = nn.Linear(num_ftrs, 2)
    return resnet18


class DQN(object):
    def __init__(self):
        # -----------Define 2 networks (target and training)------#
        self.eval_net, self.target_net = get_net().to(device), get_net().to(device)
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
        self.memory[1][index] = a
        self.memory[2][index] = r
        self.memory[3][index, :] = s_
        self.memory_counter += 1

    def choose_action(self, x):
        x = torch.from_numpy(x.copy()).to(dtype=torch.float32, device=device)
        with torch.no_grad():
            # add 1 dimension to input state x
            x = torch.unsqueeze(x, 0)
            # input only one sample
            if np.random.uniform() < EPSILON:  # greedy
                # use epsilon-greedy approach to take action
                actions_value = self.eval_net(x)
                # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
                # what we need is the index in this function, representing the
                # action of cart.
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            else:  # random
                action = np.random.randint(0, N_ACTIONS)
        return action

    def learn(self):
        self.optimizer.zero_grad()
        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_s = torch.tensor(self.memory[0][index, :]).to(dtype=torch.float32,device=device)
        b_a = torch.tensor([self.memory[1][index]]).to(dtype=torch.int64,device=device)
        b_r = torch.tensor(self.memory[2][index]).to(dtype=torch.float32,device=device)
        b_s_ = torch.tensor(self.memory[3][index, :]).to(dtype=torch.float32,device=device)

        q_eval = self.eval_net(b_s).gather(1, b_a.repeat_interleave(32, dim=0))  # (batch_size, 1)

        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * \
                   q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        loss.backward()
        self.optimizer.step()
        logging.info(loss)
#
#
# dqn = DQN()
#
# # env = FlappyBird(Comm)
#
# # Start training
# print("\nCollecting experience...")
# for i_episode in range(400):
#     state = env.reset()
#     ep_r = 0
#     while True:
#         # take action based on the current state
#         action = dqn.choose_action(state)
#         # obtain the reward and next state and some other information
#         state_next, reward, done = env.step(action)
#
#         # store the transitions of states
#         dqn.store_transition(state, action, reward, state_next)
#         ep_r += reward
#         if dqn.memory_counter > MEMORY_CAPACITY:
#             dqn.learn()
#             if done:
#                 print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
#         if done:
#             break
#         # use next state to update the current state.
#         state = state_next
