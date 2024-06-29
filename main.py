from dqn_alg import Agent
from game import Model
import numpy as np

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    env = Model()
    agent = Agent(gamma=0.9, epsilon=1.0, batch_size=64, n_actions=11,
                  eps_end=0.01, input_dims=[2], lr=0.0003)
    scores, eps_history, losses, avg = [], [], [], []


    def plot_loss(scores_, show_result=False):
        plt.figure(2)
        loss_t = torch.tensor(scores_, dtype=torch.float)
        eps_t = torch.tensor(avg, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Optimization number')
        plt.ylabel('Loss')
        plt.plot(loss_t.numpy())
        plt.plot(eps_t)

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


    n_games = 1000

    for i in range(n_games):
        score = 0
        done = 0
        observation = env.reset()
        loss_buff = []
        while done == 0:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action - 5)
            score += reward
            agent.store_transitions(observation, action, reward,
                                    observation_, done)
            loss = agent.learn()
            if loss is not None:
                loss_buff.append(loss)
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        mloss = torch.mean(torch.tensor(loss_buff))
        losses.append(mloss)

        avg_score = np.mean(scores[-100:])
        avg.append(avg_score)

        plot_loss(losses)

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon,
              'mean loss %.2f' % mloss)


    plot_loss(losses, show_result=True)

    plt.ioff()
    plt.show()

    a = np.zeros((21, 21))

    for i in range(21):
        for j in range(21):
            state = env.reset([i, j])
            action = agent.choose_action(state, random_=False)
            a[i, j] = action - 5

    print(a)
    ax = sns.heatmap(a, linewidth=0.5)
    ax.invert_yaxis()

    plt.show()

