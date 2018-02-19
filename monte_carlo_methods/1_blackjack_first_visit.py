import numpy as np

'''
the object of blackjack is to obtain cards summing up to 21 without exceeding this value
all the face cards count as 10, ace can value 1 or 11, depending on the player
in this version each player compete independently against the dealer

at the beginning of the game both player and dealer have 2 cards
one of the dealer's card is faced up

1)if the player has a face card (10) and an ace (11) this is called NATURAL
if the dealer has also a NATURAL it is a draw, otherwise the player wins

2)if the player doesn't have a NATURAL, at each turn:
he can request for 1 additional card (HITS) until he stops (STICKS) or exceed 21 (BUST)
if he goes BUST he loses, if he STICKS it becomes the dealer's turn

the dealer also HITS or STICKS according to a fixed strategy:
    if sum is lower than 17 -> STICK
    else -> HIT

MDP formulation of the game:
reward is: +1, -1, or 0 (GAMMA = 0 -> no discount)
actions: the player can HIT or STICK
states: player cards, dealer's showing card
policy: the player STICKS if sum is 20 or 21, otherwise HITS (fixed policy)

if the player holds an ace that can be evaluated 11 without going BUST:
the ace is said to be USABLE and in same scenarios it is always counted as 11
(otherwise there is no decision to make, since counting ace as 1, the sum would be <= 11)
'''

'''
monte carlo first visit: 
only value function is updated, policy is fixed
value of a state s is the average of the returns obtained after the first visit of s in each episode
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
V = np.zeros(shape=[D1,D2,D3])
R = np.zeros(shape=[D1,D2,D3])
f = np.ones(shape=[D1,D2,D3])

EPISODES = 1000
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
            # fixed policy
            state = [np.sum(player_cards)+10*usable_ace-12,dealer_cards[0]-1,usable_ace]
            if state not in states:
                states.append(state)
                f[state[0]][state[1]][state[2]] += 1
            if np.sum(player_cards)+10*usable_ace < 20:
                player_cards.append(hit(13))
            else: d = stick()
            if np.sum(player_cards) + 10 * usable_ace > 21:
                d = True
                player_bust = True
            if state[0] == 21: d = True
    # dealer's turn
    dealers_turn = True
    r = 0
    if player_bust == True:
        dealers_turn = False
        print "player bust!"
        r = -1.0
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
        if np.sum(dealer_cards) > np.sum(player_cards):
            print "dealer wins!"
            r = -1.0
        elif np.sum(dealer_cards) == np.sum(player_cards):
            print "draw!"
            r = 0.0
        else:
            print "player wins!"
            r = 1.0
    # update return for each visited state
    for s in range(len(states)):
        visited_state = states[s]
        R[visited_state[0]][visited_state[1]][visited_state[2]] += r

    # update value function summing R along episodes and dividing by frequency
    V = np.divide(R,f)
print V
