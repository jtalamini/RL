import numpy as np

# arms probability
# 10-arms bandit

k = 10
RUNS = 100
EPISODES = 10000
ALPHA = 0.1

# perform action on the environment

def pull_arm(n):
    reward = 0.0
    win_prob = np.random.uniform()
    if win_prob < arms_prob[n]: reward = 1.0
    return reward

'''
 incremental evaluation of action-value:
 t = times action a has been selected
 appropriate method for non-stationary problems: reward probability is not fixed over time
 Qt+1(a) = Qt(a) + alpha*(Rt(a) - Qt(a))
'''

def estimate(a):
    Q_ = ALPHA*(R[-1] - Q[a])
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
    '''
    optimistic initialization of action-value = 5
    this encourage exploration: whichever actions are selected at the beginning, 
    the reward is less than the initial estimate, thus switching to other actions
    
    the agent perform exploration even if a greedy policy is selected
    '''
    Q = np.ones(k)*5
    while episode < EPISODES:
        episode += 1
        # greedy action: maximize immediate reward
        arm = np.argmax(Q)
        r = pull_arm(arm)
        # increment number of times action a occurs
        R.append(r)
        pulled_arms.append(arm)
        # update action value function
        Q[arm] += estimate(arm)
    score.append(np.sum(R)/EPISODES)
    if np.argmax(arms_prob) == np.argmax(Q): accuracy += 1.0

print "Accuracy: ", accuracy/RUNS
print "Average score: ", np.mean(score)