import numpy as np
import gym

'''
Double Q-learning optimizes action-value function Q to the optimal Q* independently of the policy being followed
Double Q-learning is not affected by the maximization bias 
(due to using the same samples to determine the maximization action and estimate its value)
off-policy method
'''

env = gym.make("FrozenLake-v0")
D = env.observation_space.n
A = env.action_space.n
Q1 = np.zeros([D,A])
Q2 = np.zeros([D,A])

def e_greedy_policy(Q1, Q2, s):
    if np.random.uniform() < EPSILON:
        a = env.action_space.sample()
    else:
        Q = Q1 + Q2
        a = np.argmax(Q[s])
    return a


EPISODES = 8000
ALPHA = 0.01
GAMMA = 0.99
BATCH = 100
EPSILON = 0.1
accuracy = []
steps = []
for e in range(EPISODES):
    s = env.reset()
    d = False
    ep_r = 0
    step = 0
    while d == False:
        step+=1
        #env.render()
        # choose A from S using epsilon-greedy policy in Q1 + Q2
        a = e_greedy_policy(Q1, Q2, s)
        s1, r, d, _ = env.step(a)
        ep_r += r
        if d == True and r == 0.0: r = -1.0
        # use 2 models to compute 1) armgaxQ, 2) Q[argmaxQ] separately
        if np.random.uniform() < 0.5:
            a1 = np.argmax(Q1[s1])
            Q1[s,a] += ALPHA*(r + GAMMA*Q2[s1,a1] - Q1[s,a])
        else:
            a1 = np.argmax(Q2[s1])
            Q2[s,a] += ALPHA*(r + GAMMA*Q1[s1,a1] - Q2[s,a])

        s = s1

    accuracy.append(ep_r)
    steps.append(step)

    if e % BATCH == 0 and e > 0:
        EPSILON /= 2
        print "batch %d accuracy %f average steps %f" %(e/BATCH, np.mean(accuracy), np.mean(steps))
        accuracy = []