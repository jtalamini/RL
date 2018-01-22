import numpy as np

# arms probability
# 10-arms bandit

k = 10
RUNS = 100
EPISODES = 1000
e = 0.1

# perform action on the environment

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
    if sum_all_r == 0: return 0
    return sum_a_r/sum_all_r

run = 0
accuracy = 0.0
score = []

while run < RUNS:
    arms_prob = np.random.normal(size=10)
    run +=1
    episode = 0
    R = []
    pulled_arms = []
    Q = np.zeros(k)
    while episode < EPISODES:
        episode += 1
        # exploration action with epsilon probability
        if np.random.rand() < e:
            arm = np.random.randint(k)
        else:
            # greedy action: maximize immediate reward
            arm = np.argmax(Q)
        r = pull_arm(arm)
        R.append(r)
        pulled_arms.append(arm)
        if episode % 10 == 0:
            # update action value function
            R_array = np.array(R)
            A_array = np.array(pulled_arms)
            for i in range(k):
                Q[i] = estimate(i)
    score.append(np.sum(R)/EPISODES)
    if np.argmax(arms_prob) == np.argmax(Q): accuracy += 1.0

print "Accuracy: ", accuracy/RUNS
print "Average score: ", np.mean(score)
