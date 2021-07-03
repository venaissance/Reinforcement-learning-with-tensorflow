"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import gym
import numpy as np

from my_DQN_brain import DeepQNetwork
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,  # 2
                  n_features=env.observation_space.shape[0],  # 4
                  learning_rate=0.01,
                  e_greedy=0.9,
                  replace_target_iter=100,
                  memory_size=2000,
                  e_greedy_increment=0.001,
                  )


def training(max_episodes=100):
    total_steps = 0
    # 存放奖励值
    GLOBAL_RUNNING_R = []

    for i_episode in range(max_episodes):
        observation = env.reset()
        total_reward = 0
        while True:
            env.render()

            action = RL.choose_action(observation)
            observation_, reward_, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            RL.store_transition(observation, action, reward, observation_)
            total_reward += reward

            print('total_steps: ', total_steps)

            if total_steps > max_episodes:
                RL.learn()

            if done:
                print('episode: ', i_episode,
                      'total_reward: ', round(total_reward, 2),
                      ' epsilon: ', round(RL.epsilon, 2))
                break

            observation = observation_
            total_steps += 1

        GLOBAL_RUNNING_R.append(total_reward)
        # 画出奖励值和时间步的关系
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.title('DQN Reward')
        plt.xlabel('episodes')
        plt.ylabel('moving reward')
        plt.show()

    env.close()


training(200)
