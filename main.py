import numpy as np
import gym
import torch
import torch.optim as optim
from torch import Tensor
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

        self.rewards = []
        self.saved_log_probs = []

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
    reward_sum = 0
    D = 80 * 80
    gamma = 0.99
    i_episode = 1
    batch_size = 10

    policy = Policy()
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-3)

    while True:
        if render:
            env.render()

        cur_x = prepro(obs)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        x = Variable(torch.from_numpy(x).float()).unsqueeze(0)
        probs = policy(x)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        action = 2 if action.data[0] == 1 else 2

        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        policy.rewards.append(reward)

        if done:
            if i_episode % batch_size == 0:
                R = 0
                policy_loss = []
                rewards = []
                for r in policy.rewards:
                    R = r + gamma * R
                    rewards.insert(0, R)
                rewards = Tensor(rewards)
                rewards =\
                    (rewards - rewards.mean())\
                    / (rewards.std() + np.finfo(np.float32).eps)

                for log_prob, reward in zip(policy.saved_log_probs, rewards):
                    policy_loss.append(-log_prob * reward)

                optimizer.zero_grad()
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                optimizer.step()

                del policy.rewards[:]
                del policy.saved_log_probs[:]

            print('%dth episode reward sum: %d' % (i_episode, reward_sum))
            obs = env.reset()
            reward_sum = 0
            prev_x = None
            i_episode += 1


if __name__ == '__main__':
    main()
