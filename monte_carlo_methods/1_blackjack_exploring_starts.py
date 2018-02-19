import numpy as np

'''
monte carlo exploring starts: 
action-value funcion Q(s,a) is computed in stead of value function V(s)
policy is updated using Q function
value of Q(s,a) is the average of the returns obtained after the first visit of (s,a) in each episode
'''

# hit
def hit(number):
    # 13 type of cards from infinite deck
    card = np.random.choice(number)+1
    if card > 10: card = 10
    return card

# stick
def stick():
    return True

def initialize_game():
    p_cards = []
    p_cards.append(hit(13))
    p_cards.append(hit(9))
    usable = 0
    if 1 in p_cards and np.sum(p_cards)+10 <= 21: usable = 1
    d_cards = []
    d_cards.append(hit(13))
    d_cards.append(hit(9))
    if 1 in d_cards and np.sum(d_cards)+10 <= 21: d_cards.append(10)
    return p_cards, d_cards, usable

# initialize V function, initialize return for each state
# number of states: 10 (12 to 21 player sum) x 10 (ace or up to 10 dealer sum) x 2 (player USABLE ace) = 200 states
D1 = 10
D2 = 10
D3 = 2
A = 2
Q = np.ones(shape=[D1,D2,D3,A])/2
R = np.zeros(shape=[D1,D2,D3,A])
f = np.ones(shape=[D1,D2,D3,A])
policy = np.ones(shape=[D1,D2,D3])

EPISODES = 10000
accuracy = []
batch_size = 100
# game start
for episode in range(EPISODES):
    player_cards, dealer_cards, usable_ace = initialize_game()
    d = False
    player_bust = False
    dealer_bust = False
    # at each step of the game evaluate state
    states = []
    while d == False:
        if np.sum(player_cards)+10*usable_ace < 12:
            player_cards.append(hit(13))
        else:
            # choose action using the policy
            state = [np.sum(player_cards)+10*usable_ace-12,dealer_cards[0]-1,usable_ace]
            if episode > 1000:
                a = policy[state[0]][state[1]][state[2]]
            else:
                # exploring start unrealistic assunmption
                if np.sum(player_cards)+10*usable_ace < 20:
                    a = 1
                else: a = 0
            state.append(int(a))
            if state not in states:
                states.append(state)
                f[state[0]][state[1]][state[2]][state[3]] += 1
            if a == 1:
                player_cards.append(hit(13))
            else: d = stick()
            if np.sum(player_cards) + 10 * usable_ace > 21:
                d = True
                player_bust = True
            if state[0] == 21: d = True

    # dealer's turn
    dealers_turn = True
    r = 0
    if dealers_turn == True:
        dealers_d = False
        while dealers_d == False:
            # fixed policy
            if np.sum(dealer_cards) < 17:
                dealer_cards.append(hit(13))
                if np.sum(dealer_cards) > 21:
                    dealers_d = True
                    dealer_bust = True
            else:
                dealers_d = stick()
        if np.sum(dealer_cards) > np.sum(player_cards): r = -1.0
        elif np.sum(dealer_cards) == np.sum(player_cards): r = 0.0
        else: r = 1.0
    if r > 0:
        accuracy.append(r)
    else: accuracy.append(0)
    if episode % batch_size == 0 and episode > 0:
        print "accuracy:", np.mean(accuracy)
        accuracy = []
    # update return for each visited state
    for s in range(len(states)):
        visited_state = states[s]
        R[visited_state[0]][visited_state[1]][visited_state[2]][visited_state[3]] += r

    # update value function summing R along episodes and dividing by frequency
    Q = np.divide(R,f)
    policy = np.argmax(Q, axis=3)
