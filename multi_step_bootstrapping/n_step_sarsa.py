import gym
import numpy as np

'''
n-step sarsa
'''

env = gym.make("FrozenLake-v0")
EPISODES = 8000
D = env.observation_space.n
A = env.action_space.n
Q = np.zeros([D,A])
EPSILON = 0.1
ALPHA = 5e-4
GAMMA = 0.99
n = 3
accuracy = []
steps = []
BATCH = 100

def e_greedy_policy(Q, s):
    if np.random.uniform() < EPSILON: a = env.action_space.sample()
    else: a = np.argmax(Q[s])
    return a

def discount_rewards(r):
    sum = 0
    for i in reversed(xrange(len(r))):
        sum = r[i] + GAMMA*sum
    return sum

for episode in range(EPISODES):
    ep_r = 0
    s = env.reset()
    d = False
    tau = 0
    T = 999
    states = []
    rewards = []
    actions = []
    a = e_greedy_policy(Q, s)
    states.append(s)
    actions.append(a)
    t = 0
    while tau < T-1:
        if t < T:
            s, r, d, _ = env.step(a)
            ep_r += r
            if d == True and r == 0.0: r = -1.0
            states.append(s)
            rewards.append(r)
            if d == True: T = t+1
            else:
                a = e_greedy_policy(Q, s)
                actions.append(a)
        # tau is the time whose estimate is being updated
        tau = t-n+1
        if tau >= 0:
            G = discount_rewards(rewards[tau:min(T, tau+n)])
            if tau + n < T: G += (GAMMA**n)*Q[states[tau+n],actions[tau+n]]
            Q[states[tau],actions[tau]] += ALPHA*(G - Q[states[tau],actions[tau]])
        t+=1
    accuracy.append(ep_r)
    steps.append(t)

    if episode % BATCH == 0 and episode > 0:
        EPSILON /= 2
        print "batch %d accuracy %f average steps %f" %(episode/BATCH, np.mean(accuracy), np.mean(steps))
        accuracy = []
        steps = []








