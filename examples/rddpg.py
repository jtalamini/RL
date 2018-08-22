from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
import random

EPISODES = 50000
A_RANGE = 2
TAU = 0.001
MEMORY_SIZE = 6000000
BATCH = 64
GAMMA = 0.99
UNITS = 300
TRAJECTORY = 5
UPDATE_FREQ = 5


class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, traj):
        if self.count < self.buffer_size:
            self.buffer.append(traj)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(traj)

    def size(self): return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else: batch = random.sample(self.buffer, batch_size)
        return np.array(batch)

    def clear(self):
        self.buffer.clear()
        self.count = 0

# takes the current state as input
# the output is a single real value: action chosen from continuous action space
class Actor:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None,D])
            self.h1 = layers.fully_connected(self.x, num_outputs=400)
            self.h2 = layers.fully_connected(self.h1, num_outputs=400)
            self.trainLength = tf.placeholder(dtype=tf.int32)
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

            self.x_3d = tf.reshape(self.h2, [self.batch_size, self.trainLength, 400])
            # time dependencies
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            self.state_init = lstm_cell.zero_state(self.batch_size, tf.float32)
            c_in = tf.placeholder(shape=[None, lstm_cell.state_size.c], dtype=tf.float32)
            h_in = tf.placeholder(shape=[None, lstm_cell.state_size.h], dtype=tf.float32)
            self.state_in = (c_in, h_in)
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, self.x_3d, initial_state=state_in)
            self.state_out = lstm_state
            rnn_out = tf.reshape(lstm_output, [-1, 256])

            self.critic_grad = tf.placeholder(dtype=tf.float32, shape=[None, A])

            self.h3 = layers.fully_connected(inputs=rnn_out, num_outputs=400)
            self.h4 = layers.fully_connected(inputs=self.h3, num_outputs=300)
            self.action = layers.fully_connected(inputs=self.h4, num_outputs=A, activation_fn=None)
            self.scaled_action = tf.nn.tanh(self.action)
            self.vars = tf.trainable_variables(name)
            # Combine the gradients here
            self.optimizer = tf.train.AdamOptimizer(1e-3)
            divisor = (tf.cast(self.batch_size, tf.float32)*tf.cast(self.trainLength, dtype=tf.float32))
            self.loss = (-self.critic_grad * self.scaled_action + tf.reduce_mean(
                tf.square(tf.layers.flatten(self.action))) * 1e-3) / divisor

            self.actor_gradient = self.optimizer.compute_gradients(loss=self.loss, var_list=self.vars)
            self.actor_gradient = [(tf.clip_by_norm(grad, 0.5), var) for grad, var in self.actor_gradient]

            self.train_step = self.optimizer.apply_gradients(self.actor_gradient)

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
            self.input = tf.concat([self.x, self.a], axis=1)

            self.h1 = layers.fully_connected(inputs=self.input, num_outputs=400)
            self.h2 = layers.fully_connected(inputs=self.h1, num_outputs=300)
            self.Q = layers.fully_connected(inputs=self.h2, num_outputs=1, activation_fn=None)
            self.loss = tf.reduce_mean(tf.square(self.y - self.Q))
            self.optimizer = tf.train.AdamOptimizer(1e-3)
            self.gradient = self.optimizer.compute_gradients(loss=self.Q, var_list=self.a)

            self.vars = tf.trainable_variables(name)
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.vars)
            for i, (grad, var) in enumerate(self.gradients):
                self.gradients[i] = (tf.clip_by_norm(grad, 0.5), var)

            self.train_step = self.optimizer.apply_gradients(self.gradients)

            self.main = tf.trainable_variables(scope="critic_main")
            self.target = tf.trainable_variables(scope="critic_target")
            self.update_target_critic_params = \
                [self.target[i].assign(tf.multiply(self.main[i], TAU) +
                                   tf.multiply(self.target[i], 1. - TAU))
                 for i in range(len(self.target))]


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
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

# MAIN
load = False
D = 3
A = 1

actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(A))
sess = tf.InteractiveSession()

actor = Actor("actor_main")
actor_target = Actor("actor_target")
critic = Critic("critic_main")
critic_target = Critic("critic_target")

sess.run(tf.global_variables_initializer())
update_target(actor_target, critic_target)
saver = tf.train.Saver()

STEP = 0
EPISODE = 0
EP_REW = 0
BATCH_REW = []
BEST_REW = 0.0

#reset environment
env = gym.make("Pendulum-v0")
s = env.reset()
lstm_states = (np.zeros([1,256]),np.zeros([1,256]))
traj_buffer = []
buffer = ReplayBuffer(buffer_size=MEMORY_SIZE)

while (EPISODE < EPISODES):
    STEP += 1
    a, lstm_states = sess.run([actor.scaled_action, actor.state_out], feed_dict={
        actor.x: np.reshape(s, [1, D]),
        actor.state_in[0]: lstm_states[0],
        actor.state_in[1]: lstm_states[1],
        actor.batch_size: 1,
        actor.trainLength: 1
    })
    a = a[0]
    a += actor_noise()
    s1, r, d, _ = env.step(a)
    EP_REW += r
    # add to memory
    traj_buffer.append([s,a,r,d,s1])

    if STEP % TRAJECTORY == 0:
        buffer.add(traj_buffer)
        traj_buffer = []

    if buffer.count > 10000 and STEP % UPDATE_FREQ == 0:
        sample = buffer.sample_batch(BATCH)

        actor_state = (np.zeros([BATCH,256]),np.zeros([BATCH,256]))

        sample = np.reshape(sample, [BATCH * TRAJECTORY, 5])

        sample_s = np.vstack(sample[:,0])
        sample_a = np.vstack(sample[:,1])
        sample_r = sample[:,2]
        sample_d = sample[:,3]
        sample_s1 = np.vstack(sample[:,4])

        # calculate target actions
        actor_target_actions = sess.run(actor_target.scaled_action, feed_dict={
            actor_target.x: sample_s1,
            actor_target.state_in[0]: actor_state[0],
            actor_target.state_in[1]: actor_state[1],
            actor_target.batch_size: BATCH,
            actor_target.trainLength: TRAJECTORY
        })
        # calculate target Q value for actions
        critic_target_q = sess.run(critic_target.Q, feed_dict={
            critic_target.x: sample_s1,
            critic_target.a: actor_target_actions,
        })
        critic_target_q = np.reshape(critic_target_q, [-1])
        critic_y = sample_r + GAMMA*critic_target_q*(1-sample_d)
        critic_y = np.reshape(critic_y, [BATCH*TRAJECTORY,1])
        # update critic
        sess.run(critic.train_step, feed_dict={
            critic.x: sample_s,
            critic.a: sample_a,
            critic.y: critic_y
        })

        # update actor
        policy = sess.run(actor.scaled_action, feed_dict={
            actor.x: sample_s,
            actor.state_in[0]: actor_state[0],
            actor.state_in[1]: actor_state[1],
            actor.batch_size: BATCH,
            actor.trainLength: TRAJECTORY
        })

        critic_grads = sess.run(critic.gradient, feed_dict={
            critic.a: policy,
            critic.x: sample_s
        })[0][0]
        sess.run(actor.train_step, feed_dict={
            actor.x: sample_s,
            actor.critic_grad: critic_grads,
            actor.state_in[0]: actor_state[0],
            actor.state_in[1]: actor_state[1],
            actor.batch_size: BATCH,
            actor.trainLength: TRAJECTORY
        })
        # update target nets
        update_target(actor_target, critic_target)

    s = s1

    if (d == True):
        lstm_states = (np.zeros([1, 256]), np.zeros([1, 256]))
        traj_buffer = []
        EPISODE += 1
        print "episode ", EPISODE, " reward: ", EP_REW
        EP_REW = 0
        if EPISODE < EPISODES:
            s = env.reset()
        STEP = 0
        if EPISODE % 100 == 0:
            print "model saved"
