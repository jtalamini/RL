import numpy as np
import gym

'''
Q-learning optimizes action-value function Q to the optimal Q* independently of the policy being followed
off-policy method
'''

env = gym.make("FrozenLake-v0")
D = env.observation_space.n
A = env.action_space.n
Q = np.zeros([D,A])

def e_greedy_policy(Q, s):
    if np.random.uniform() < EPSILON:
        a = env.action_space.sample()
    else: a = np.argmax(Q[s])
    return a


EPISODES = 8000
ALPHA = 0.5
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
        a = e_greedy_policy(Q, s)
        s1, r, d, _ = env.step(a)
        ep_r += r
        if d == True and r == 0.0: r = -1.0
        # choose a1 from s1 using arg max Q(s1)
        a1 = np.argmax(Q[s1])
        Q[s,a] += ALPHA*(r + GAMMA*Q[s1,a1] - Q[s,a])
        s = s1

    accuracy.append(ep_r)
    steps.append(step)

    if e % BATCH == 0 and e > 0:
        EPSILON /= 2
        print "batch %d accuracy %f average steps %f" %(e/BATCH, np.mean(accuracy), np.mean(steps))
        accuracy = []