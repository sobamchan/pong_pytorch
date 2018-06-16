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
                nn.Linear(200, 3),
                nn.Sigmoid(),
                ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Agent:

    def __init__(self, policy, optimizer, params):
        self.policy = policy
        self.optimizer = optimizer
        self.params = params
        self.rewards = []
        self.saved_log_probs = []

    def select_action(self, state):
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.data[0] + 1

    def learn(self):
        params = self.params
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards:
            R = r + params['gamma'] * R
            rewards.insert(0, R)
        rewards = Tensor(rewards)
        rewards = (rewards - rewards.mean())\
            / (rewards.std() + np.finfo(np.float32).eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]


def prepro(img):
    img = img[35:195]  # Crop
    img = img[::2, ::2, 0]  # Downsample by factor of 2
    img[img == 144] = 0  # Erase background
    img[img == 109] = 0  # Erase background
    img[img != 0] = 1  # Erase background
    return img.astype(np.float).ravel()


def main():
    render = False

    env = gym.make("Pong-v0")  # action: 2->up, 3->down
    obs = env.reset()
    prev_x = None
    reward_sum = 0
    D = 80 * 80
    params = {
            'gamma': 0.99,
            'decay_rate': 0.99,
            'batch_size': 3,
            }

    i_episode = 1
    running_reward = None

    policy = Policy()
    optimizer = optim.RMSprop(policy.parameters(),
                              lr=1e-3,
                              weight_decay=params['decay_rate'])
    agent = Agent(policy, optimizer, params)

    while True:
        if render:
            env.render()

        cur_x = prepro(obs)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        x = Variable(torch.from_numpy(x).float()).unsqueeze(0)
        action = agent.select_action(x)

        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        agent.rewards.append(reward)

        if done:
            if i_episode % params['batch_size'] == 0:
                agent.learn()

            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = 0.01 * reward_sum + 0.99 * running_reward
            print('%dth episode reward sum: %d, mean reward: %f'
                  % (i_episode, reward_sum, running_reward))
            obs = env.reset()
            reward_sum = 0
            prev_x = None
            i_episode += 1
        if reward != 0:
            print('episode done, reward: %d' % reward)


if __name__ == '__main__':
    main()
