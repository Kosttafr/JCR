from scipy.stats import poisson
import numpy as np


class Model:
    def __init__(self, state=np.random.choice(21, 2)):
        self.state = state

    def reset(self, state=np.random.choice(21, 2)):
        self.state = state

    def apply_action(self, action):
        return np.array([max(min(self.state[0] - action, 20), 0),
                         max(min(self.state[1] + action, 20), 0)])

    def step(self, action):
        reward = 0
        new_state = self.apply_action(action)

        reward += (-2) * abs(action)

        a_request = poisson.rvs(mu=3)
        a_return = poisson.rvs(mu=3)
        b_request = poisson.rvs(mu=4)
        b_return = poisson.rvs(mu=2)

        valid_a_request = min(new_state[0], a_request)
        valid_b_request = min(new_state[1], b_request)

        reward += (valid_a_request + valid_b_request) * 10

        new_state = np.array([max(min(new_state[0] - valid_a_request + a_return, 20), 0),
                              max(min(new_state[1] - valid_b_request + b_return, 20), 0)])

        time_frame = self.state, action, new_state, reward
        self.state = new_state
        return time_frame

    def modify_state(self, state):
        new_state = np.zeros(21 * 21)
        new_state[21 * self.state[1] + self.state[0]] = 1
        return new_state

# c = Model()
# print(c.step(3))
# print(c.state)
# print(np.array([1, 2]))
# print(poisson.rvs(mu=3))
# print(poisson.rvs(mu=3))
# print(np.random.choice(2, p=[0.9, 0.1]))
# print(np.random.choice(11) - 5)
