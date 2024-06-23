import math
import random
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import count

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import game
import nnet
import memory as mem

BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.005
LR = 1e-4

OPTIMIZATION_FREQUENCY = 5
MAX_TIME_OF_PLAY = 16

REAL_TIME_LOSS_RENDER = True

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

env = game.Model()

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

n_actions = 11
field_size = 21

policy_net = nnet.DQN(field_size * field_size, n_actions).to(device)
target_net = nnet.DQN(field_size * field_size, n_actions).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR)

memory = mem.ReplayMemory(100000)

steps_done = 0


def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    sample = np.random.choice(2, p=[eps_threshold, 1 - eps_threshold])
    steps_done += 1
    if sample == 1:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            sigma = policy_net(torch.tensor(env.modify_state(state), dtype=torch.float32, device=device).unsqueeze(0))

            # print(sigma)

            max_act_val = None
            max_act = None

            t12 = min(state[0], 5)
            t21 = -min(state[1], 5)

            for act in range(5 + t21, 5 + t12 + 1):
                # print(act)
                # print(sigma[act])
                if max_act_val is None:
                    max_act_val = sigma[0][act]
                    max_act = act - 5
                elif max_act_val < sigma[0][act]:
                    max_act_val = sigma[0][act]
                    max_act = act - 5

            return torch.tensor(np.array(max_act), device=device, dtype=torch.long)
    else:
        return torch.tensor(np.array(np.random.choice(11) - 5), device=device, dtype=torch.long)


episode_loss = []


def plot_loss(show_result=False):
    plt.figure(2)
    loss_t = torch.tensor(episode_loss, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Optimization number')
    plt.ylabel('Loss')
    plt.plot(loss_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.random_sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = mem.Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action) + 5
    reward_batch = torch.cat(batch.reward)

    next_state_batch = torch.cat(batch.next_state)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    with torch.no_grad():
        next_state_buff = target_net(state_batch)
        next_state_values = next_state_buff.max(1)[0]
        # print(next_state_buff)
        # a = input()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    if next_state_batch[0] is None:
        expected_state_action_values = reward_batch[0].unsqueeze(0)
    else:
        expected_state_action_values = ((next_state_values[0] * GAMMA) + reward_batch[0]).unsqueeze(0)

    for i in range(1, len(next_state_values)):
        if next_state_batch[i] is None:
            expected_state_action_values = torch.cat((expected_state_action_values, reward_batch[i].unsqueeze(0)), 0)
        else:
            expected_state_action_values = torch.cat(
                (expected_state_action_values, ((next_state_values[i] * GAMMA) + reward_batch[i]).unsqueeze(0)), 0)

    # Compute Mean Square loss
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    episode_loss.append(loss)

    # print(loss)
    if REAL_TIME_LOSS_RENDER:
        plot_loss()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def dqn_mod(ep_num):
    if torch.cuda.is_available():
        num_episodes = 6000
    else:
        num_episodes = ep_num

    g_reward = 0
    for i_episode in range(num_episodes):
        if i_episode % 20 == 0:
            print('episode №' + str(i_episode))
            if len(episode_loss) > 0:
                print("current loss: " + str(episode_loss[len(episode_loss) - 1]))
        # Initialize the environment and get it's state
        env.reset(state=np.array([(i_episode // 21) % 21, i_episode % 21]))

        g_reward = 0
        for t in count():
            action = select_action(env.state)

            state, action, new_state, reward = env.step(action)

            state = torch.tensor(env.modify_state(state), dtype=torch.float32, device=device).unsqueeze(0)

            action = action.reshape(1, 1)

            # print(action)
            # a = input()

            reward = torch.tensor([reward], device=device)

            next_state = torch.tensor(env.modify_state(new_state), dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            optimize_model()

            if t % OPTIMIZATION_FREQUENCY == 0:
                # Perform one step of the optimization (on the policy network)
                # optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

            if t > MAX_TIME_OF_PLAY:
                break

            # print(t)


dqn_mod(200)

# dqn(1000)

print('Complete')
plot_loss(show_result=True)

plt.ioff()
plt.show()

f = open('text.txt', 'w')
def save_policy():
    for i in range(21):
        for j in range(21):
            env.reset(state=np.array([i, j]))
            state = torch.tensor(env.modify_state(env.state), dtype=torch.float32, device=device).unsqueeze(0)
            # state = torch.tensor(env.state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = policy_net(state).max(1)[1].view(1, 1) - int((n_actions - 1)/2)
                print(policy_net(state))

            f.write(str(a) + '  ')
        f.write('\n')

    f.close()
