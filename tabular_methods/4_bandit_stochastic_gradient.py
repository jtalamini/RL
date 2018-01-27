import numpy as np

# arms probability
# 10-arms bandit

k = 10
RUNS = 100
EPISODES = 10000
e = 0.1
alpha = 0.1

# perform action on the environment

def pull_arm(n):
    reward = 0.0
    win_prob = np.random.uniform()
    if win_prob < arms_prob[n]: reward = 1.0
    return reward

run = 0
accuracy = 0.0
score = []

while run < RUNS:
    arms_prob = np.random.normal(size=10)
    run +=1
    episode = 0
    R = []
    pulled_arms = []
    H = np.zeros(k)
    policy = np.zeros(k)
    while episode < EPISODES:
        episode += 1
        if np.random.rand() < e:
            arm = np.random.randint(k)
        else:
            '''
            compute policy as soft-max of preference H
            '''
            policy = np.exp(H)/np.sum(np.exp(H))
            arm = np.argmax(policy)
        y = np.zeros(k)
        y[arm] = 1
        r = pull_arm(arm)
        R.append(r)
        pulled_arms.append(arm)
        '''
        update preference function H
        stochastic gradient ascent:
        each action preference H(a) is incremented proportionally to its effect on performances dE(R)/dH
        '''
        H += alpha * (R[-1] - np.mean(R)) * (y - policy)
    score.append(np.sum(R)/EPISODES)
    if np.argmax(arms_prob) == np.argmax(H): accuracy += 1.0

print "Accuracy: ", accuracy/RUNS
print "Average score: ", np.mean(score)
