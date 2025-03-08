# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        ppo_td.py
# Author:           wzw
# Version:          0.1
# Created:          2025/3/6
# Description:      重要性采样,策略步伐控制,on policy,采用了值函数
#                   步骤1:初始化环境,设置一个记忆模块
#                   步骤2:三个模型,一个policy,一个target,一个value
#                   步骤3:使用policy玩游戏,随机挑选动作,并且记录到记忆模块
#                   步骤4:如果游戏结束,则更新target
# ------------------------------------------------------------------



import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GAMMA = 0.95  # discount factor
LR_ACTOR = 1e-3 # learning rate
LR_CRITIC=1e-2 # learning rate
# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# torch.backends.cudnn.enabled = False  # 非确定性算法


class ActorTarget(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorTarget, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out
class ActorBehavior(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorBehavior, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out
# ----------------------------------- #
# 构建价值网络--critic
# ----------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class PPO(object):
    def __init__(self, env):
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.ep_states, self.ep_actions, self.ep_rewards, self.ep_old_logprob,self.ep_next_states,self.done= [], [], [], [],[],[]

        # 目标策略
        self.actor_target = ActorTarget(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.actor_behavior = ActorBehavior(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.critic=ValueNet(n_states=self.state_dim,n_hiddens=20).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_target.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)



    def choose_action_behavior(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        behavior_output = self.actor_behavior.forward(observation)
        with torch.no_grad():
            prob_behavior = F.softmax(behavior_output, dim=0).detach()
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(prob_behavior)
        # 依据其概率随机挑选一个动作
        action = action_list.sample()
        return action.item(), action_list.log_prob(action)

    def choose_action_test(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        target_output = self.actor_target.forward(observation)
        with torch.no_grad():
            prob_target = F.softmax(target_output, dim=0).data.cpu().numpy()
        action = np.argmax(prob_target)
        return action

    # 将状态,动作,奖励,策略,下步状态,完成标志保存到六个列表里
    def store_transition(self, states, actions, rewards, old_logprob,next_states,done):
        self.ep_states.append(states)
        self.ep_actions.append(actions)
        self.ep_rewards.append(rewards)
        self.ep_old_logprob.append(old_logprob)
        self.ep_next_states.append(next_states)
        self.done.append(done)

    def update(self):
        # Step 1: 预测当前步的价值
        q_states_target_predict=torch.squeeze(self.critic(torch.FloatTensor(self.ep_states).to(device))).cpu().detach().numpy()
        # Step 2: 预测下一步的价值
        q_next_states_target_predict=torch.squeeze(self.critic(torch.FloatTensor(self.ep_next_states).to(device))).cpu().detach().numpy()
        # Step 3: 估算当前步真实价值
        q_state_target_true=q_next_states_target_predict*GAMMA*(1-np.array(self.done,dtype=np.float32))+np.asarray(self.ep_rewards,dtype=np.float32)
        # Step 3: 计算折扣优势
        discounted_ep_rs=q_state_target_true-q_states_target_predict
        discounted_ep_rs = torch.tensor(discounted_ep_rs, dtype=torch.float).to(device)

        # discounted_ep_rs /= torch.std(discounted_ep_rs)  # 除以标准差
        for iter in range(10):
            logits = self.actor_target.forward(torch.FloatTensor(self.ep_states).to(device))
            neg_log_prob = F.cross_entropy(input=logits, target=torch.LongTensor(self.ep_actions).to(device),
                                           reduction='none')
            actor_loss = torch.mean(-torch.clamp(torch.exp(-neg_log_prob - torch.tensor(self.ep_old_logprob).to(device)), 0.8,
                                           1.2) * discounted_ep_rs)
            states_predict=torch.squeeze(self.critic(torch.FloatTensor(self.ep_states).to(device)))
            critic_loss=torch.mean(nn.functional.mse_loss(states_predict,torch.tensor(q_state_target_true).to(device)))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 每次学习完后清空数组
        self.ep_states, self.ep_actions, self.ep_rewards, self.ep_old_logprob ,self.ep_next_states,self.done= [], [], [], [],[],[]


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = PPO(env)
    # 游戏盘数
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()[0]
        # Train
        # 一局游戏玩多少步
        agent.actor_behavior.load_state_dict(agent.actor_target.state_dict())
        for step in range(STEP):
            action, old_logprob = agent.choose_action_behavior(state)  # softmax概率选择action
            next_state, reward, done, _, info = env.step(action)
            agent.store_transition(state, action, reward, old_logprob,next_state,done)  # 新函数 存取这个transition
            state = next_state
            if done:
                # print("stick for ",step, " steps")
                agent.update()  # 更新策略网络
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()[0]
                for j in range(STEP):
                    env.render()
                    action = agent.choose_action_test(state)  # direct action for test
                    state, reward, done, _, info = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
