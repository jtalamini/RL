import gym
import numpy as np
from gym.envs.registration import register
register(
    id='FrozenLakeDeterministic-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

env = gym.make("FrozenLakeDeterministic-v0")
D = env.observation_space.n
A = env.action_space.n
Q = np.zeros([D,A])
model = {"s": np.zeros([D,A])-1, "r": np.zeros([D,A])-1}

'''
Initialize empty prioritized queue
if P(s,a) is greater than the threshold -> (s,a) is put into the queue with priority P
samples from the model are picked according to the priority
'''
P_queue = {}
THETA = 0.5

def e_greedy_policy(Q, s):
    if np.random.uniform() < EPSILON:
        a = env.action_space.sample()
    else:
        a = np.argmax(Q[s])
    return a

EPISODES = 8000
ALPHA = 0.001
GAMMA = 0.99
BATCH = 100
EPSILON = 0.5
n = 5
visited_s = []
visited_a = [[]]*D

accuracy = []
steps = []
for episode in range(EPISODES):
    s = env.reset()
    d = False
    step = 0
    ep_r = 0
    while d == False:
        step += 1
        a = e_greedy_policy(Q, s)
        s1, r, d, _ = env.step(a)
        ep_r += r
        if d == True and r == 0.0: r = -1.0
        '''
        compute priority for (s,a)
        '''
        a1 = np.argmax(Q[s1])
        P = np.abs(r + GAMMA*Q[s1,a1] - Q[s,a])
        if (P > THETA):
            P_queue[(s,a)] = P

        # model update
        if s not in visited_s: visited_s.append(s)
        if a not in visited_a[s]: visited_a[s] = visited_a[s] + [a]
        model["s"][s,a] = s1
        model["r"][s,a] = r

        for i in range(n):
            if (len(P_queue.keys()) > 0):
                id = np.argmax(P_queue.values())
                sv, av = P_queue.keys()[id]
                del P_queue[(sv, av)]
                sm1 = model["s"][sv,av]
                rm = model["r"][sv,av]
                Q[sv,av] += ALPHA*(rm + GAMMA*np.max(Q[int(sm1)]) - Q[sv, av])
                # we update the queue according to the model of the env
                # by adding to the queue the couples (s,a) that are likely to lead to sv
                for i in range(D):
                    for j in range(A):
                        if model["s"][i][j] == sv:
                            prev_s = i
                            prev_a = j
                            prev_r = model["r"][prev_s, prev_a]
                            P = np.abs(prev_r + GAMMA * np.max(Q[sv]) - Q[prev_s, prev_a])
                            if P > THETA:
                                P_queue[(prev_s, prev_a)] = P

        s = s1

    accuracy.append(ep_r)
    steps.append(step)

    if (episode % BATCH == 0 and episode > 0):
        EPSILON /= 2
        print "batch %d accuracy %f average steps %f" %(episode/BATCH, np.mean(accuracy), np.mean(steps))
        accuracy = []
        steps = []