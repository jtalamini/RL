import numpy as np

# arms probability
# 10-arms bandit

k = 10
RUNS = 100
EPISODES = 10000
e = 0.1

# perform action on the environment

def pull_arm(n):
    reward = 0.0
    win_prob = np.random.uniform()
    if win_prob < arms_prob[n]: reward = 1.0
    return reward

'''
 incremental evaluation of action-value:
 t = times action a has been selected
 appropriate method for stationary problems: reward probability is fixed over time
 Qt+1(a) = 1/t * sum(R)
 ...
 Qt+1(a) = Qt(a) + 1/t*(Rt(a) - Qt(a))
'''

def estimate(a):
    Q_ = (R[-1] - Q[a]) / N[a]
    return Q_

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
    N = np.zeros(k)
    while episode < EPISODES:
        episode += 1
        # exploration action with epsilon probability
        if np.random.rand() < e:
            arm = np.random.randint(k)
        else:
            # greedy action: maximize immediate reward
            arm = np.argmax(Q)
        r = pull_arm(arm)
        # increment number of times action a occurs
        N[arm] += 1
        R.append(r)
        pulled_arms.append(arm)
        if episode % 10 == 0:
            # update action value function
            Q[arm] += estimate(arm)
    score.append(np.sum(R)/EPISODES)
    if np.argmax(arms_prob) == np.argmax(Q): accuracy += 1.0

print "Accuracy: ", accuracy/RUNS
print "Average score: ", np.mean(score)