import numpy as np

'''
4x4 gridworld, terminal states are (0,0) and (3,3)
in each state movements are allowed in each direction
'''
D = 4
A = 4
THETA = 1e-20

# 1. Initialization
V = np.zeros(shape=[D, D])
PI = np.ones(shape=[D * D]) / A

# 2. Policy Evaluation
def policy_evaluation():
    while True:
        DELTA = 0
        for s in range(D * D):
            x = s / D
            y = s % D
            v = V[x][y]
            values, rewards = evaluate_states(x, y)
            d = False
            # no state can be reached from terminal states
            if s == 0 or s == 15: d = True
            if d == False:
                V[x][y] = np.max(rewards + values)
                '''
                faster evaluation using max value 
                '''
                DELTA = max(DELTA, abs(v - V[x][y]))
        if DELTA < THETA:
            return

# 3. Policy Improvement
def policy_improvement():
    for s in range(D*D):
        old_action = np.argmax(PI[s])
        x = s % D
        y = s / D
        values, rewards = evaluate_states(x,y)
        PI[s] = np.argmax(rewards + values)

def move(x, y, direction):
    if direction==0:
        y = max(0, y-1)
    elif direction==1:
        x = min(D-1, x+1)
    elif direction==2:
        y = min(D-1, y+1)
    else:
        x = max(0, x-1)
    return x, y

def evaluate_states(x_,y_):
    values = []
    rewards = []
    for i in range(4):
        x, y = move(x_,y_,i)
        r = -1.0
        values.append(V[x][y])
        rewards.append(r)
    return np.array(values), np.array(rewards)


'''
here value iteration is used: 
1. policy evaluation: one step to compute value function using max (less computation required)
2. policy improvement: update of policy
'''

policy_evaluation()
policy_improvement()
print V

'''
show results
'''

# convert policy values into directions
def num_to_str(list):
    res = []
    for i in range(len(list)):
        s = "left"
        if list[i] == 0: s = "up"
        elif list[i] == 1: s = "right"
        elif list[i] == 2: s = "down"
        res.append(s)
    return res

m = num_to_str(PI)
m = np.reshape(m, newshape=[D,D])
m[0,0] = "x"
m[3,3] = "x"
print m