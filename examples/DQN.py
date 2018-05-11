import gym
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle
import time

D = 84
SEQ = 4
A = 4
GAMMA = 0.99
MEMORY_SIZE = 2e5
SAMPLE = 32
EPSILON = 1.0
annealing_steps = 1000000.
stepDrop = (0.9) / annealing_steps
tau = 0.001
TAU = 0.001
LOAD = False


# model definition
class Model:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, D, D, SEQ])
            self.Q_target = tf.placeholder(dtype=tf.float32, shape=[None])
            self.a_pl = tf.placeholder(dtype=tf.int32, shape=[None])

            self.h1 = layers.conv2d(inputs=self.x, num_outputs=16, kernel_size=8, stride=4)
            self.h2 = layers.conv2d(inputs=self.h1, num_outputs=32, kernel_size=4, stride=2)
            self.h2_flat = layers.flatten(self.h2)
            self.fc1 = layers.fully_connected(inputs=self.h2_flat, num_outputs=256)
            self.Q = layers.fully_connected(inputs=self.fc1, num_outputs=A, activation_fn=None)
            # q-learning
            self.a_indexes = tf.one_hot(indices=self.a_pl, depth=A, dtype=tf.float32)
            self.loss = tf.reduce_mean(
                tf.square(tf.reduce_sum(tf.multiply(self.Q, self.a_indexes), axis=1) - self.Q_target))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, epsilon=0.01, decay=0.95)
            self.train_step = self.optimizer.minimize(self.loss)


model = Model("model")
target = Model("target")

model_vars = tf.trainable_variables("model")
target_vars = tf.trainable_variables("target")
update_target = [target_vars[i].assign(tf.multiply(model_vars[i], TAU) + tf.multiply(target_vars[i], 1. - TAU)) for i in
                 range(len(target_vars))]


def preprocess(frame):
    frame = np.uint8(resize(rgb2gray(frame), (110, 84), mode="constant") * 255)
    frame = frame[19:103, :]
    return frame


def sample_memory(memory, size):
    sublist = np.random.choice(xrange(len(memory["s"])), size)
    s_samp = np.float32([memory["s"][i] for i in sublist]) / 255.0
    a_samp = [memory["a"][i] for i in sublist]
    r_samp = [memory["r"][i] for i in sublist]
    d_samp = [memory["d"][i] for i in sublist]
    s1_samp = np.float32([memory["s1"][i] for i in sublist]) / 255.0
    return s_samp, a_samp, r_samp, d_samp, s1_samp


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


EPISODE = 0
FRAMES = []
MEMORY = {"s": [], "a": [], "r": [], "d": [], "s1": []}
BATCH_SCORE = []
TENBATCH = []
BEST = 0
SCORE = 0
LIVES = 5
STEP = 0
FIRED = False
score_dict = {"episode": [], "score": [], "best": [], "epsilon": []}

env = gym.make("BreakoutDeterministic-v4")
s = preprocess(env.reset())
FRAMES = [s] * SEQ

sess = tf.InteractiveSession()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
if (LOAD == True):
    saver.restore(sess, "DQN/DQN.ckpt")
    score_dict = pickle.load(open("score.p", "rb"))
    EPISODE = score_dict["episode"][len(score_dict["episode"]) - 1]
    EPSILON = score_dict["epsilon"][len(score_dict["epsilon"]) - 1]

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

while True:
    STEP += 1
    # env.render()
    # time.sleep(0.1)
    p = np.random.uniform()
    if (p < EPSILON):
        action = env.action_space.sample()
    else:
        f = np.reshape(FRAMES, [-1, D, D, SEQ]).astype(float) / 255.0
        action = np.argmax(sess.run(model.Q, feed_dict={model.x: f}))
    if action == 1: FIRED = True
    s1, r, d, info = env.step(action)
    if r > 0: SCORE += r
    dead = 0
    r = np.sign(r)
    if info["ale.lives"] < LIVES:
        r = -1
        LIVES -= 1
        dead = 1
        STEP = 0
        FIRED = False
    if (STEP >= 30 and FIRED == False):
        dead = 1
        d = True
        r = -1
    # process next frames
    s1 = preprocess(s1)
    # memory
    if (len(MEMORY["s"]) == MEMORY_SIZE):
        MEMORY = {j: MEMORY[j][SAMPLE:] for j in MEMORY.iterkeys()}
    MEMORY["s"].append(np.reshape(FRAMES, [D, D, SEQ]))
    MEMORY["a"].append([action])
    MEMORY["r"].append([r])
    MEMORY["d"].append([dead])
    FRAMES.remove(FRAMES[0])
    FRAMES.append(s1)
    MEMORY["s1"].append(np.reshape(FRAMES, [D, D, SEQ]))

    if EPISODE > 100 and len(MEMORY["s"]) >= MEMORY_SIZE // 10:
        # decrease epsilon-greedy
        if EPSILON > 0.1: EPSILON -= stepDrop
        if STEP % 4 == 0:
            # update main model
            sample_s, sample_a, sample_r, sample_d, sample_s1 = sample_memory(MEMORY, SAMPLE)
            sample_s1_Q = np.reshape(sample_r, [SAMPLE]) + GAMMA * np.reshape(
                np.max(sess.run(target.Q, feed_dict={target.x: sample_s1}), axis=1), [SAMPLE]) * (
                                      1 - np.reshape(sample_d, [SAMPLE]))
            sess.run(model.train_step, feed_dict={model.x: sample_s, model.Q_target: sample_s1_Q,
                                                  model.a_pl: np.reshape(sample_a, [SAMPLE])})
            # update target model
            sess.run(update_target)
            # updateTarget(targetOps, sess)
    if d == True:
        EPISODE += 1
        FIRED = False
        STEP = 0
        LIVES = 5
        BATCH_SCORE.append(SCORE)
        BEST = max(SCORE, BEST)
        SCORE = 0
        FRAMES = []
        s = preprocess(env.reset())
        FRAMES = [s] * SEQ

        if EPISODE % 10 == 0:
            # print stats
            print "batch mean score: %f best score %d" % (np.mean(BATCH_SCORE), BEST)
            TENBATCH.append(np.mean(BATCH_SCORE))
            BATCH_SCORE = []

        if EPISODE % 100 == 0:
            score_dict["episode"].append(EPISODE)
            score_dict["score"].append(np.mean(TENBATCH))
            score_dict["best"].append(BEST)
            score_dict["epsilon"].append(EPSILON)
            TENBATCH = []

            print "Random action: ", EPSILON
            # save session
            save_path = saver.save(sess, "DQN/DQN.ckpt")
            print "Episode %d\tModel saved!" % (EPISODE)
            pickle_out = open("score.p", "wb")
            pickle.dump(score_dict, pickle_out)
            pickle_out.close()
