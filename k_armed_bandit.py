import numpy as np

# arms probability
# 6-arms bandit
arms_prob = [0.3, 0.1, 0.5, 0.0, 0.1, 0.0]
k = 6

def pull_arm(n):
    reward = 0.0
    win_prob = np.random.rand()
    if win_prob < arms_prob[n]: reward = 1.0
    return reward

'''
 action-value estimate:
 sample-average method
 Qt(a) = sum of reward for action a prior to t / sum of reward for all actions prior to t
'''

def estimate(a):
    indexes = np.argwhere(A_array==a)
    a_r = R_array[indexes]
    sum_a_r = float(np.sum(a_r))
    sum_all_r = float(np.sum(R_array))
    return sum_a_r/sum_all_r

# exploration
e = 1.0
episode = 0
R = []
pulled_arms = []
Q = np.zeros(k)

while episode < 1000:
    episode += 1
    if np.random.rand() < e:
        arm = np.random.randint(k)
    else:
        arm = np.argmax(Q)
    r = pull_arm(arm)
    R.append(r)
    pulled_arms.append(arm)
    if episode % 10 == 0:
        # update exploitation-exploration
        e = max(e-0.1,0)
        # update arms values
        R_array = np.array(R)
        A_array = np.array(pulled_arms)
        for i in range(k):
            Q[i] = estimate(i)
        print Q
