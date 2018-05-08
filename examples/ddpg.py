import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque
import random
import numpy as np
import gym
import time

D = 3
A = 1
A_RANGE = 2
TAU = 0.001
BUFF_SIZE = 1e6
BATCH = 64
GAMMA = 0.99

class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s1):
        experience = (s,a,r,d,s1)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self): return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else: batch = random.sample(self.buffer, batch_size)
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        d_batch = np.array([_[3] for _ in batch])
        s1_batch = np.array([_[4] for _ in batch])
        return s_batch,a_batch,r_batch,d_batch,s1_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


# takes the current state as input
# the output is a single real value: action chosen from continuous action space
class Actor:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None,D])
            self.critic_grad = tf.placeholder(dtype=tf.float32, shape=[None, A])
            self.h1 = layers.fully_connected(inputs=self.x, num_outputs=400)
            self.h2 = layers.fully_connected(inputs=self.h1, num_outputs=300)
            self.action = layers.fully_connected(inputs=self.h2, num_outputs=A, activation_fn=tf.nn.tanh)
            self.scaled_action = tf.multiply(self.action, A_RANGE)
            self.vars = tf.trainable_variables(name)
            # Combine the gradients here
            self.actor_unnorm_gradient = tf.gradients(self.scaled_action, self.vars, -self.critic_grad)
            self.actor_gradient = list(map(lambda x: tf.div(x, BATCH), self.actor_unnorm_gradient))
            self.optimizer = tf.train.AdamOptimizer(0.0001)
            self.train_step = self.optimizer.apply_gradients(zip(self.actor_gradient, self.vars))

            self.main = tf.trainable_variables(scope="actor_main")
            self.target = tf.trainable_variables(scope="actor_target")
            self.update_target_actor_params = \
                [self.target[i].assign(tf.multiply(self.main[i], TAU) +
                                   tf.multiply(self.target[i], 1. - TAU))
                 for i in range(len(self.target))]


# compute the Q value of the current states and actions chosen by the actor
class Critic:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None,D])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None,1])
            self.a = tf.placeholder(dtype=tf.float32, shape=[None,A])

            self.h1 = layers.fully_connected(inputs=self.x, num_outputs=400)
            self.h2 = layers.fully_connected(inputs=self.h1, num_outputs=300, activation_fn=None, biases_initializer=None)
            self.h3 = layers.fully_connected(inputs=self.a, num_outputs=300, activation_fn=None)
            self.h4 = tf.nn.relu(self.h3 + self.h2)
            self.Q = layers.fully_connected(inputs=self.h4, num_outputs=1, activation_fn=None)
            self.loss = tf.reduce_mean(tf.square(self.y - self.Q))
            self.gradient = tf.gradients(self.Q, self.a)
            self.optimizer = tf.train.AdamOptimizer(0.001)
            self.train_step = self.optimizer.minimize(self.loss)

            self.main = tf.trainable_variables(scope="critic_main")
            self.target = tf.trainable_variables(scope="critic_target")
            self.update_target_critic_params = \
                [self.target[i].assign(tf.multiply(self.main[i], TAU) +
                                   tf.multiply(self.target[i], 1. - TAU))
                 for i in range(len(self.target))]


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def update_target(aaa, ccc):
    sess.run(aaa.update_target_actor_params)
    sess.run(ccc.update_target_critic_params)

actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(A))

actor = Actor("actor_main")
actor_target = Actor("actor_target")
critic = Critic("critic_main")
critic_target = Critic("critic_target")
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
update_target(actor_target, critic_target)
buffer = ReplayBuffer(BUFF_SIZE)
saver = tf.train.Saver()

load = False

if load:
    saver.restore(sess, "ddpg/ddpg.ckpt")

env = gym.make("Pendulum-v0")
s = env.reset()
STEP = 0
EPISODE = 0
EP_REW = 0
while True:
    STEP += 1
    if load:
        env.render()
        time.sleep(0.1)
    a = sess.run(actor.scaled_action, feed_dict={actor.x: np.reshape(s, [1,D])}) + actor_noise()
    s1, r, d, info = env.step(a[0])
    EP_REW += r
    # add to memory
    buffer.add(np.reshape(s,[D]), np.reshape(a,[A]), r, d, np.reshape(s1,[D]))
    if buffer.size() > BATCH:
        sample_s, sample_a, sample_r, sample_d, sample_s1 = buffer.sample_batch(BATCH)
        # calculate target actions
        actor_target_actions = sess.run(actor_target.scaled_action, feed_dict={actor_target.x: sample_s1})
        # calculate target Q value for actions
        critic_target_q = sess.run(critic_target.Q, feed_dict={
            critic_target.x: sample_s1,
            critic_target.a: actor_target_actions
        })
        critic_target_q = np.reshape(critic_target_q, [-1])
        critic_y = sample_r + GAMMA*critic_target_q*(1-sample_d)
        critic_y = np.reshape(critic_y, [BATCH,1])
        # update critic
        sess.run(critic.train_step, feed_dict={critic.x: sample_s, critic.a: sample_a, critic.y: critic_y})
        # update actor
        policy = sess.run(actor.scaled_action, feed_dict={actor.x: sample_s})
        critic_grads = sess.run(critic.gradient, feed_dict={
            critic.a: policy,
            critic.x: sample_s,
        })
        sess.run(actor.train_step, feed_dict={actor.x: sample_s, actor.critic_grad: critic_grads[0]})
        # update target nets
        update_target(actor_target, critic_target)

    s = s1

    if d == True:
        EPISODE += 1
        print "episode %d\treward %f" %(EPISODE, EP_REW)
        EP_REW = 0
        s = env.reset()
        STEP = 0
        if EPISODE % 100 == 0:
            saver.save(sess, "ddpg/ddpg.ckpt")