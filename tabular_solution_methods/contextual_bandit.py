import numpy as np

# arms probability
# 10-arms bandit

k = 4
BANDITS = 5
EPISODES = 10000
e = 0.1
alpha = 0.1
arms_prob = [[0.2,0.5,0.1,0.2],[0.05,0.7,0.05,0.2],[0.4,0.2,0.2,0.2],[0.5,0.2,0.2,0.1],[0.3,0.1,0.5,0.1]]

'''
this is an associative search task, because it involves both: 
1) trial-and-error learning for best actions
2) association of actions with situations in which are the best

this is not a full reinforcement learning problem:
actions affect only immediate reward, not next situation
'''

# pick random bandit

def random_pick_bandit():
    return np.random.choice(BANDITS)

# perform action on the environment

def pull_arm(bandit, n):
    reward = 0.0
    win_prob = np.random.rand()
    if win_prob < arms_prob[bandit][n]: reward = 1.0
    return reward

episode = 0
R = []
H = np.zeros([BANDITS,k])
policy = np.zeros(k)
while episode < EPISODES:
    episode += 1
    bandit = random_pick_bandit()
    policy = np.exp(H[bandit]) / np.sum(np.exp(H[bandit]))
    if np.random.rand() < e:
        arm = np.random.randint(k)
    else:
        '''
        compute policy as soft-max of preference H
        '''
        arm = np.argmax(policy)
    y = np.zeros(k)
    y[arm] = 1
    r = pull_arm(bandit,arm)
    R.append(r)
    '''
    update preference function H
    stochastic gradient ascent:
    each action preference H(a) is incremented proportionally to its effect on performances dE(R)/dH
    '''
    H[bandit] += alpha * (R[-1] - np.mean(R)) * (y - policy)
print "Predicted best arm for each bandit: ", np.argmax(H, axis=1)
print "True best arm for each bandit:", np.argmax(arms_prob, axis=1)