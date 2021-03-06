import argparse
from itertools import count
from collections import namedtuple

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--decay-rate', type=float, default=0.99)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('-seed', type=int, default=0)
    return parser.parse_args()


def prepro(img):
    img = img[35:195]  # Crop
    img = img[::2, ::2, 0]  # Downsample by factor of 2
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    return img.astype(np.float).ravel()


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.li = nn.Linear(6400, 200)
        self.action_head = nn.Linear(200, 3)
        self.value_head = nn.Linear(200, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.li(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


class Agent:

    def __init__(self, args):
        self.args = args

        self.policy = Policy()
        self.optimizer = optim.RMSprop(self.policy.parameters(),
                                       lr=args.learning_rate,
                                       weight_decay=args.decay_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.policy(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_actions.append(SavedAction(m.log_prob(action),
                                                     state_value))
        return action.data[0]

    def finish_episode(self):
        args = self.args
        R = 0
        policy_losses = []
        value_losses = []
        rewards = []
        saved_actions = self.policy.saved_actions

        for r in self.policy.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean())\
            / (rewards.std() + np.finfo(np.float32).eps)

        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.data[0]
            policy_losses.append(-log_prob.data * reward)
            value_losses.append(F.smooth_l1_loss(value,
                                                 Variable(torch.Tensor([r]))))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum()\
            + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()

        del self.policy.rewards[:]
        del self.policy.saved_actions[:]


def main():
    args = get_args()
    env = gym.make('Pong-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = Agent(args)

    running_reward = None
    reward_sum = 0
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):
            state = prepro(state)
            action = agent.select_action(state)
            action = action + 1
            state, reward, done, _ = env.step(action)
            reward_sum += reward

            agent.policy.rewards.append(reward)

            if done:
                if running_reward is None:
                    running_reward = reward_sum
                else:
                    running_reward = running_reward * 0.99 + reward_sum * 0.01
                print('resetting env. reward: %d, mean reward: %f'
                      % (reward_sum, running_reward))
                reward_sum = 0
                break

            if reward != 0:
                print('episode %d: game. reward: %f' % (i_episode, reward))

        if i_episode % args.batch_size == 0:
            print('episode %d: updating network' % i_episode)
            agent.finish_episode()


if __name__ == '__main__':
    main()
