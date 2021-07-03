import gym
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()


class ActorNetwork(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], name='state')
        self.a = tf.placeholder(tf.int32, None, name='act')
        self.td_error = tf.placeholder(tf.float32, None, name='td_error')  # TD_error
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # 隐藏层单元
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.1),  # 权重
                bias_initializer=tf.constant_initializer(0.1),  # bias
                name='l1'
            )
            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # 输出的单元
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.1),  # 权重
                bias_initializer=tf.constant_initializer(0.1),  # bias
                name='acts_prob'
            )

            # Loss function
            with tf.variable_scope('exp_v'):
                log_prob = tf.log(self.acts_prob[0, self.a])  # 平方差
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # 时间差分误差
            with tf.variable_scope('train'):
                # self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict={
            self.s: s,
            self.a: a,
            self.td_error: td
        })
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        # get all actions' probabilities
        probes = self.sess.run(self.acts_prob, {self.s: s})
        # 返回一个动作值
        # Error: act return as a TUPLE, not a int
        act = np.random.choice(probes.shape[1], p=probes.ravel()),
        return act


class CriticNetwork(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], name='state')
        self.v_ = tf.placeholder(tf.float32, [1, 1], name='v_next')  # next state's Q
        self.r = tf.placeholder(tf.float32, None, name='r')  # reward
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # 隐藏层单元
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.1),  # 权重
                bias_initializer=tf.constant_initializer(0.1),  # bias
                name='l1'
            )

            # 设定v值包含的参数
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # 输出单元
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.1),  # 权重
                bias_initializer=tf.constant_initializer(0.1),  # bias
                name='V'
            )

            # calc TD_Error = (r + gamma * V_next) - V_eval and Loss
            with tf.variable_scope('squared_TD_error'):
                self.td_error = self.r + GAMMA * self.v_ - self.v
                self.loss = tf.square(self.td_error)
            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})  # get next state's Q
        td_error, _ = self.sess.run([self.td_error, self.train_op], feed_dict={
            self.s: s,
            self.v_: v_,
            self.r: r
        })
        return td_error


# Training#############################################

MAX_EPISODES = 200  # 最大回合
MAX_EP_STEPS = 200  # 每回合的最大时间步
GAMMA = 0.9
LR_A = 0.001  # actor-net learning rate
LR_C = 0.01  # critic-net learning rate
env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped
# 环境中的状态与动作空间
N_F = env.observation_space.shape[0]
N_A = env.action_space.n
sess = tf.Session()
actor = ActorNetwork(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = CriticNetwork(sess, n_features=N_F, lr=LR_C)
sess.run(tf.global_variables_initializer())


def training(max_episodes=MAX_EPISODES):
    for i_episode in range(max_episodes):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            env.render()
            act = actor.choose_action(s)
            s_, r, done, info = env.step(act)

            if done:
                r = -20
            track_r.append(r)
            td_error = critic.learn(s, r, s_)
            actor.learn(s, act, td_error)
            s = s_
            t += 1
            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                print('Episode: %i' % i_episode, "|Reward: %i " % int(running_reward))
                break

        # 画出奖励值和时间步的关系
        plt.plot(np.arange(len(track_r)), track_r)
        plt.title('Actor-Critic Reward')
        plt.xlabel('episodes')
        plt.ylabel('moving reward')
        plt.show()

    env.close()


training()
