import numpy as np
import gym
import torch
import torch.optim as optim
from torch import Tensor
# from torch import LongTensor
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions import Categorical


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        layers = [
                nn.Linear(6400, 200),
                nn.ReLU(),
                nn.Linear(200, 1),
                nn.Sigmoid(),
                ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def prepro(img):
    img = img[35:195]  # Crop
    img = img[::2, ::2, 0]  # Downsample by factor of 2
    img[img == 144] = 0  # Erase background
    img[img == 109] = 0  # Erase background
    img[img != 0] = 1  # Erase background
    return img.astype(np.float).ravel()


def discount_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = 0 if r[t] != 0 else running_add
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    render = False

    env = gym.make("Pong-v0")  # action: 2->up, 3->down
    obs = env.reset()
    prev_x = None
    # running_reward = None
    reward_sum = 0
    episode_number = 0
    D = 80 * 80

    policy = Policy()
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-3)

    while True:
        rewards = []
        if render:
            env.render()

        cur_x = prepro(obs)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        x = Variable(torch.from_numpy(x).float()).unsqueeze(0)
        probs = policy(x)
        m = Categorical(probs)
        action = m.sample()
        action = 2 if action == 1 else 2
        dlogps.append(m.log_prob(action))

        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        rewards.append(reward)

        if done:
            episode_number += 1
