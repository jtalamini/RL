import numpy as np

'''
4x4 gridworld, terminal states are (0,0) and (3,3)
in each state movements are allowed in each direction
'''
D = 4
A = 4
THETA = 1e-20

'''
equiprobable random policy: all actions are equally likely
the goal is to compute the value function for each state using this policy
this helps us finding better policies
'''

V = np.zeros(shape=[D,D])

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
        s1 = y*D + x
        if s1 == 0 or s1 == 15:
            r = 0.0
        values.append(V[x][y])
        rewards.append(r)
    return np.array(values), np.array(rewards)

while True:
    DELTA = 0
    for s in range(D*D):
        x = s/D
        y = s % D
        v = V[x][y]
        values, rewards = evaluate_states(x,y)
        d = False
        # no state can be reached from terminal states
        if s == 0 or s == 15: d = True
        if d == False:
            V[x][y] = np.mean(rewards + values)
            DELTA = max(DELTA,abs(v - V[x][y]))
    print V
    if DELTA < THETA: break
