# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        dqn.py
# Author:           wzw
# Version:          0.1
# Created:          2025/3/6
# Description:      off policy,经验回放,用于离散
#                   步骤1:初始化环境,设置一个记忆模块
#                   步骤2:两个模型,一个policy,一个target,policy选择动作所在的索引的值,target选择Q值最大的值
#                   步骤3:每一步更新policy,使用平滑L1损失
#                   步骤4:如果游戏结束,则更新target
# ------------------------------------------------------------------
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

# 设置 matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# 是否GPU被使用
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """存储一个转换"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class DQN(nn.Module):

    # 状态维度,动作维度,这里的动作其实就是Q值
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE是从重放缓冲区中采样的转换次数
# GAMMA是前一节中提到的折扣因子
# EPS_START是epsilon的起始值
# EPS_END是epsilon的最终值
# EPS_DECAY控制epsilon的指数衰减速率，越高表示衰减越慢
# TAU是目标网络的更新速率
# LR是“AdamW”优化器的学习率
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# 获取动作维度
n_actions = env.action_space.n
# 获取状态
state, info = env.reset()
# 获取状态维度
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    # 随机生成一个随机数
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # 大于eps_threshold则采用Q值选Q值最大的动作
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1)将返回每行的最大列值。
            # 选择具有更大预期奖励的行动
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 取100集的平均值，并把它们画出来
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # 随机取出一个batch的数据
    transitions = memory.sample(BATCH_SIZE)
    # 批处理
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接批处理元素
    # (最终状态应该是模拟结束后的状态)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 预测的当前状态Q值
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算所有下一个状态的V（s_{t+1}）
    # non_final_next_states的动作期望值是基于计算的
    # 在“旧的”target_net上；选择最大(1)个值的最佳奖励
    # 这是基于掩码合并的，这样我们就会有预期的
    # state值，如果状态为final，则为0。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # 估计当前状态Q值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 平滑L1损失,防止梯度爆炸
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 防止梯度爆炸
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # 初始化环境
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        # terminated:终止 truncated:截断
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 存储状态
        memory.push(state, action, next_state, reward)

        # 移动到下一状态
        state = next_state

        # 每走一步更新一次
        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()