import numpy as np

'''
4x4 gridworld, terminal states are (0,0) and (3,3)
in each state movements are allowed in each direction
'''
D = 4
A = 4
GAMMA = 0.9
V = np.zeros(shape=[D,D])

'''
equiprobable random policy: all actions are equally likely
the goal is to obtain the value function for each state
'''

def random_policy():
    return np.random.choice(A)

def move(x, y, direction):
    if direction==0:
        y = max(0, y-1)
    elif direction==1:
        x = min(D-1, x+1)
    elif direction==2:
        y = min(D-1, y+1)
    else:
        x = max(0, x-1)
    s1 = y*D + x
    if s1 == 0 or s1 == 15: d == True
    return x, y, -1.0, d

episodes = 1000
for ep in range(episodes):
    x = np.random.choice(D)
    y = np.random.choice(D)
    d = False
    while d == False:
        a = random_policy()
        x1, y1, r, d = move(x,y,a)
        V[x][y] = r + GAMMA*V[x][y]
    print V

