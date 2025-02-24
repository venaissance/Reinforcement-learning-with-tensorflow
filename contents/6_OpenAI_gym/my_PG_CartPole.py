import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if (tf.__version__ >= '2'):
    import tensorflow._api.v2.compat.v1 as tf

    tf.disable_v2_behavior()


class Memory(object):
    def __init__(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []

    def store_transition(self, obs, act, rwd):
        self.ep_obs.append(obs)
        self.ep_act.append(act)
        self.ep_rwd.append(rwd)

    def convert_to_array(self):
        # 将(N,)的array转为垂直的vertical: [1,2,3] => [[1], [2], [3]]
        array_obs = np.vstack(self.ep_obs)
        array_act = np.array(self.ep_act)
        array_rwd = np.array(self.ep_rwd)
        return array_obs, array_act, array_rwd

    def reset(self):
        self.ep_obs, self.ep_act, self.ep_rwd = [], [], []


class ActorNetwork(object):
    def __init__(self, act_dim, name):
        self.act_dim = act_dim
        self.name = name

    def step(self, obs, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(obs, 10, tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            act_prob = tf.layers.dense(h1, self.act_dim, None,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            return act_prob

    def choose_action(self, obs, reuse=False):
        act_prob = self.step(obs, reuse)
        all_act_prob = tf.nn.softmax(act_prob, name='act_prob')
        return all_act_prob

    def get_neglogp(self, obs, act, reuse=True):
        act_prob = self.step(obs, reuse)
        neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_prob, labels=act)
        return neglogp


class PolicyGradient(object):
    def __init__(self, act_dim, obs_dim, lr, gamma):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.gamma = gamma
        self.OBS = tf.placeholder(tf.float32, [None, self.obs_dim], name='observation')
        self.ACT = tf.placeholder(tf.int32, [None, ], name='action')
        self.RWD = tf.placeholder(tf.float32, [None, ], name='discounted_reward')

        # get Actor and memory
        actor = ActorNetwork(self.act_dim, 'actor')
        self.memory = Memory()

        # choose action and calc loss
        self.action = actor.choose_action(self.OBS)
        neglogp = actor.get_neglogp(self.OBS, self.ACT)
        loss = tf.reduce_mean(neglogp * self.RWD)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # initialize
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]
        prob_weights = self.sess.run(self.action, feed_dict={self.OBS: obs})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        print('action', prob_weights, action)
        return action

    def learn(self):
        obs, act, rwd = self.memory.convert_to_array()
        discounted_rwd = self.discount_and_norm_rewards(rwd)
        self.sess.run(self.optimizer, feed_dict={self.OBS: obs,
                                                 self.ACT: act,
                                                 self.RWD: discounted_rwd})
        self.memory.reset()

    def discount_and_norm_rewards(self, rwd):
        discounted_rwd = np.zeros_like(rwd)
        running_add = 0
        for t in reversed(range(0, len(rwd))):
            running_add = running_add * self.gamma + rwd[t]
            discounted_rwd[t] = running_add
        print('discounted_rwd', discounted_rwd)
        discounted_rwd -= np.mean(discounted_rwd)
        discounted_rwd /= np.std(discounted_rwd)
        return discounted_rwd


env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

PG = PolicyGradient(act_dim=env.action_space.n,
                    obs_dim=env.observation_space.shape[0],
                    lr=0.02,
                    gamma=0.99)

GLOBAL_RUNNING_R = []


def training(max_episodes=100):
    for i_episode in range(max_episodes):
        obs0 = env.reset()
        ep_rwd = 0
        while True:
            env.render()
            act = PG.step(obs0)
            obs1, rwd, done, _ = env.step(act)

            PG.memory.store_transition(obs0, act, rwd)

            obs0 = obs1
            ep_rwd += rwd

            if done:
                PG.learn()
                print('Ep: %i' % i_episode, "|Ep_r: %i " % ep_rwd)
                break

        GLOBAL_RUNNING_R.append(ep_rwd)
        # 画出奖励值和时间步的关系
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.title('Policy Gradient Reward')
        plt.xlabel('episodes')
        plt.ylabel('moving reward')
        plt.show()

    env.close()


training()
