import numpy as np

'''
for some states we would like to know whether or not we should change policy
to deterministically choose an action a != PI(s).
one way to answer this is to consider selecting a in s and thereafter following 
the existing policy PI.

Qpi(s,pi'(s)) = Epi'[Rt+1 + GAMMA*Vpi(St+1)|St=s]

the value of this behaviour is the Q function:
if Q(s,a) >= Vpi(s) -> it is better to select a in s and thereafter follow PI:
it is better to select a every time in s 

POLICY IMPROVEMENT
the greedy policy is now:

PI'(s) = argmax_a(Qpi(s,a))

the greedy policy takes the action that looks best in the short term according to Vpi.
the greedy policy is as good as or better than the original policy

iterative policy:
evaluation of PI -> VPI -> improvement of PI -> PI' -> evaluation of PI' -> VPI' -> improvement of PI' -> PI''

in a finite MDP the policy converges to an optimal policy in a finite number of steps 
'''


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
                V[x][y] = np.mean(rewards + values)
                DELTA = max(DELTA, abs(v - V[x][y]))
        if DELTA < THETA:
            return

# 3. Policy Improvement
def policy_improvement():
    stable = True
    for s in range(D*D):
        old_action = np.argmax(PI[s])
        x = s % D
        y = s / D
        values, rewards = evaluate_states(x,y)
        PI[s] = np.argmax(rewards + values)
        if old_action != np.argmax(PI[s]):
            stable = False
    return stable

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

optimal = False
while optimal == False:
    policy_evaluation()
    optimal = policy_improvement()

print V
print PI

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