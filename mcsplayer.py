from pypokerengine.players import BasePokerPlayer
from time import sleep, perf_counter as pf
from random import shuffle, choice as rchoice
import pprint
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card

class MCSPlayer(BasePokerPlayer):
  
    def __init__(self):
        super().__init__()
        self.street = None
        self.imsmallblind = None

    def declare_action(self, valid_actions, hole_card, round_state):
        # print(round_state['action_histories'])
        # print(round_state)

        street = round_state['street']
        seats = round_state['seats']
        pot = round_state['pot']['main']['amount']

        community_cards = round_state['community_card'] 
        my_id = round_state['next_player']
        imsmallblind = True if my_id == round_state['small_blind_pos'] else False
        history = round_state['action_histories'][street]
        smallblind_history = [history[i] for i in range(0, len(history), 2)]
        bigblind_history = [history[i] for i in range(1, len(history), 2)]

        my_history, opponent_history = (smallblind_history, bigblind_history) if imsmallblind else (bigblind_history, smallblind_history)
        self.imsmallblind = imsmallblind

        # print(f'my_history: {my_history}')
        # print(f'opponent_history: {opponent_history}')

        smallblind_bet = smallblind_history[-1]['amount'] if len(smallblind_history) > 0 else 0
        bigblind_bet = bigblind_history[-1]['amount'] if len(bigblind_history) > 0 else 0

        shistory = [item['action'].lower() for item in smallblind_history]
        bhistory = [item['action'].lower() for item in bigblind_history]

        try:
            pot -= history[-1]['amount']
        except IndexError:
            pass
        
        try:
            pot -= history[-2]['amount']
        except IndexError:
            pass
        assert pot >= 0

        # print('pot', pot)
        # print(history, pot)
        # assert shistory[0] != 'bigblind'
        # assert bhistory[0] != 'smallblind'

        try:
            if shistory[0] == 'smallblind':
                shistory.pop(0)
            if bhistory[0] == 'bigblind':
                bhistory.pop(0)
        except IndexError:
            pass

        # print(shistory, smallblind_bet)
        # print(bhistory, bigblind_bet)
        # print(self.street)

        actions = [a['action'] for a in valid_actions]
        scores  = {'fold': 0, 'raise': 0, 'call': 0}
        # print(actions)
        # print(actions)

        num_simulations = 100000
        # timer = pf()
        for action in actions:
            
            if self.imsmallblind:
                my_history_trial = shistory + [action]
                opponent_history_trial = bhistory.copy()

                if action == 'call':
                    sbet_trial = bigblind_bet
                    bbet_trial = bigblind_bet
                if action == 'raise':
                    sbet_trial = (bigblind_bet + 20) if street == 'preflop' or street == 'flop' else (bigblind_bet + 40)
                    bbet_trial = bigblind_bet
                if action == 'fold':
                    sbet_trial = smallblind_bet
                    bbet_trial = bigblind_bet

                for i in range(num_simulations):
                    winner, payout = self.run_montecarlo_simulation(
                        current_street=street,
                        shist_this_street=my_history_trial,
                        bhist_this_street=opponent_history_trial,
                        sbet_so_far=sbet_trial,
                        bbet_so_far=bbet_trial,
                        hole_cards=hole_card,
                        community_cards=community_cards,
                        sraisecount=self.getraisecounts(round_state['action_histories'], 'smallblind'),
                        braisecount=self.getraisecounts(round_state['action_histories'], 'bigblind'),
                        pot=pot
                    )

                    if winner == 'smallblind':
                        scores[action] += payout
                    else:
                        scores[action] -= payout
                    
            else:
                my_history_trial = bhistory + [action]
                opponent_history_trial = shistory.copy()

                if action == 'call':
                    bbet_trial = smallblind_bet
                    sbet_trial = smallblind_bet
                if action == 'raise':
                    bbet_trial = (smallblind_bet + 20) if street == 'preflop' or street == 'flop' else (smallblind_bet + 40)
                    sbet_trial = smallblind_bet
                if action == 'fold':
                    bbet_trial = bigblind_bet
                    sbet_trial = smallblind_bet

                for i in range(num_simulations):
                    winner, payout = self.run_montecarlo_simulation(
                        current_street=street,
                        shist_this_street=opponent_history_trial,
                        bhist_this_street=my_history_trial,
                        sbet_so_far=sbet_trial,
                        bbet_so_far=bbet_trial,
                        hole_cards=hole_card,
                        community_cards=community_cards,
                        sraisecount=self.getraisecounts(round_state['action_histories'], 'smallblind'),
                        braisecount=self.getraisecounts(round_state['action_histories'], 'bigblind'),
                        pot=pot
                    )

                    if winner == 'bigblind':
                        scores[action] += payout
                    else:
                        scores[action] -= payout

        # timer = pf() - timer
        # print(f'simulation time: {timer}')    
        # exit(0)
        # print(scores)

        # get action with max score
        k = list(scores.keys())
        v = list(scores.values())
        m = max(v)
        best_action = k[v.index(m)]

        return best_action

        # print(round_state)
        # print(f'id: {my_id} | street: {street} | pot: {pot} | hole: {hole_card} | comm: {community_cards} | smallblind: {imsmallblind}')

        for a in valid_actions:
            if a['action'] == 'raise': return 'raise'

        return 'call'

    def getraisecounts(self, history, player):
        assert player == 'smallblind' or player == 'bigblind'

        streets_to_check = []
        raise_count = 0
        for street in history.keys():
            street_hist = history[street]
            if len(street_hist) == 0: continue

            if player == 'smallblind':
                player_hist = [street_hist[i] for i in range(0, len(street_hist), 2)]
            else:
                player_hist = [street_hist[i] for i in range(1, len(street_hist), 2)]
            
            player_hist = [item['action'].lower() for item in player_hist]
            raise_count += player_hist.count('raise')
        
        return raise_count





    def receive_game_start_message(self, game_info):
          pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        self.street = street
        self.raise_amt = 40 if (street == 'turn' or street == 'river') else 20
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
    
    def run_montecarlo_simulation(self, 
                                current_street: str, 
                                shist_this_street: list['str'], 
                                bhist_this_street: list[str], 
                                sbet_so_far: int, 
                                bbet_so_far: int, 
                                hole_cards: list[str], 
                                community_cards: list[str], 
                                sraisecount: int, 
                                braisecount: int, 
                                pot: int):
        
        # print('shist_this_street:', shist_this_street)
        
        # to prevent weird reference manipulation
        shist_this_street = shist_this_street.copy()
        bhist_this_street = bhist_this_street.copy()
        hole_cards = hole_cards.copy()
        community_cards = community_cards.copy()
    
        streets_to_simulate = ['preflop', 'flop', 'turn', 'river']

        while current_street != streets_to_simulate[0]:
            streets_to_simulate.pop(0)

        # sample cards
        samp = get_sample_card_draw(hole_cards, community_cards)
        opponent_hole, remaining_community_cards = samp[:2], samp[2:]
        community_sample = community_cards + remaining_community_cards
        # debug
        # print(f'my hole: {hole_cards}')
        # print(f'generated opponent hole: {opponent_hole}')
        # print(f'generated community: {community_sample}')

        for street in streets_to_simulate:
            
            # debug
            # print('STREET:', street)

            if street == current_street:
                shist = shist_this_street
                bhist = bhist_this_street
                sbet = sbet_so_far
                bbet = bbet_so_far
            else:
                shist = []
                bhist = []
                sbet = 0
                bbet = 0

            preflop = True if street == 'preflop' else False

            raise_amount = 20 if street == 'preflop' or street == 'flop' else 40

            street_result = run_street_simulation(
                shist=shist, 
                bhist=bhist, 
                sbet=sbet, 
                bbet=bbet, 
                preflop=preflop, 
                raise_amount=raise_amount,
                sraisecount=sraisecount,
                braisecount=braisecount
            )
            # debug
            # print(street_result)

            game_ended = street_result[0]
            if game_ended == True:
                ... # game end logic here
                who_won = street_result[1]
                assert who_won == 0 or who_won == 1
                pot += street_result[2]

                winner = 'smallblind' if who_won == 0 else 'bigblind'
                return winner, pot

            # game did not end, process street result and move on
            pot += street_result[1]
            sraisecount = street_result[4]
            braisecount = street_result[5]

        # showdown time!
        if self.imsmallblind:
            winner = self.eval_result(smallblind_hole=hole_cards, bigblind_hole=opponent_hole, community=community_sample)
        else:
            winner = self.eval_result(smallblind_hole=opponent_hole, bigblind_hole=hole_cards, community=community_sample)
        
        return winner, pot
    
    def eval_result(self, smallblind_hole, bigblind_hole, community):
        assert len(smallblind_hole) == 2
        assert len(bigblind_hole) == 2
        assert len(community) == 5

        smallblind_hole = [Card.from_str(c) for c in smallblind_hole]
        bigblind_hole = [Card.from_str(c) for c in bigblind_hole]
        community = [Card.from_str(c) for c in community]

        handevaluator = HandEvaluator()
        smallblind_score = handevaluator.eval_hand(smallblind_hole, community)
        bigblind_score   = handevaluator.eval_hand(bigblind_hole, community)

        if smallblind_score == bigblind_score:
            # print('Warning: equal score resolving deterministically towards bigblind')
            ...

        if smallblind_score > bigblind_score:
            return  'smallblind'
        
        return 'bigblind'
        


    

def setup_ai():
    return MCSPlayer()

def run_street_simulation(shist = [], bhist = [], sbet = 10, bbet = 20, preflop = True, raise_amount=20, sraisecount = 0, braisecount = 0):
    pot = 0
    # print(shist)
    while True:
        valid_moves, whos_turn = get_valid_street_moves(shist, bhist, sbet, bbet, preflop=preflop)
        assert whos_turn == 0 or whos_turn == 1

        current_hist = (shist, bhist)[whos_turn]
        opponent_hist = (shist, bhist)[(whos_turn + 1) % 2]


        if whos_turn == 0:
            if sraisecount == 4:
                valid_moves = remove_raise(valid_moves)
        
        else:
            if braisecount == 4:
                valid_moves = remove_raise(valid_moves)

        if valid_moves == None:
            assert get_last_move(current_hist) != 'fold'
            if get_last_move(opponent_hist) == 'fold':
                return True, whos_turn, sbet + bbet, shist, bhist # format: game won, who won, payout, move histories
            
            assert get_last_move(opponent_hist) == 'call'
            assert sbet == bbet

            pot += sbet + bbet
            break
        
        move = pick_move(valid_moves)

        current_hist.append(move)

        if move == 'fold':
            # do nothing, included for completeness
            ...

        if move == 'call':
            if whos_turn == 0:
                sbet = bbet
            else:
                bbet = sbet
        
        if move == 'raise':
            if whos_turn == 0:
                sbet = bbet + raise_amount
                sraisecount += 1
            else:
                bbet = sbet + raise_amount
                braisecount += 1

    # format: game not won, current pot amount, move histories, smallblind raises, bigblind raises
    return False, pot, shist, bhist, sraisecount, braisecount 

def pick_move(moves):

    # if 'raise' in moves:
    #     return 'raise'
    # elif 'call' in moves:
    #     return 'call'
    # else:
    #     raise ValueError('Huh')

    return rchoice(moves)
    exclude_fold = moves.copy()
    exclude_fold.remove('fold')
    return rchoice(exclude_fold)
        
def remove_raise(moves):
    if moves == None:
        return None

    newmoves = moves.copy()
    if 'raise' in moves:
        newmoves.remove('raise')
    
    return newmoves if len(newmoves) != 0 else None        


def get_valid_street_moves(shistory: list[str], bhistory: list[str], sbet: int, bbet: int, preflop: bool):

    '''
    Given player histories and bet amounts, return the possible moves the player can make

    ARGUMENTS

    shistory: the move history of the small blind player within this street
    bhistory: the move history of the big blind player within this street
    sbet:     the current cumulative bet amount of the small blind player in this street
    bbet:     the current cumulative bet amount of the big blind player in this street
    preflop:  if this street is a preflop (matters for the 4 raise per street rule)

    RETURNS

    list[str] containing allowed moves of the player if there are any OR
    None if there are no allowed moves and the street is over 

    IMPORTANT
    This function accounts for the within-street raise rule (the total number of raises within a street across players cannot be more than 4), 
    but not the within-round raise rule (the total number of raises of any one player in a ground cannot be more than 4)
    '''

    whos_turn = 0 if len(shistory) == len(bhistory) else 1
    
    if whos_turn == 0 and sbet > bbet and bhistory[-1] != 'fold':
        assert False, f'Something wrong here, {whos_turn}, sbet: {sbet}, bbet: {bbet}, shistory: {shistory}, bhistory: {bhistory}'

    if whos_turn == 1 and sbet < bbet and shistory[-1] != 'fold':
        assert False, f'Something wrong here too, whos_turn: {whos_turn}, sbet: {sbet}, bbet: {bbet}, shistory: {shistory}, bhistory: {bhistory}'

    allowed = ['fold', 'call', 'raise']
    
    current_history = (shistory, bhistory)[whos_turn]
    opponent_history = (shistory, bhistory)[(whos_turn + 1) % 2]

    if get_last_move(opponent_history) == 'fold':
        return None, whos_turn

    how_many_raises = (current_history+opponent_history).count('raise')
    allowed_raises = 3 if preflop else 4

    if how_many_raises >= allowed_raises:
        allowed.remove('raise')
    
    if get_last_move(current_history) == 'call' and get_last_move(opponent_history) == 'call':
        return None, whos_turn
    
    if get_last_move(current_history) == 'raise' and get_last_move(opponent_history) == 'call':
        return None, whos_turn
    
    # if get_last_move(current_history) == 'call' and get_last_move(opponent_history) == 'raise':
    #     return None

    return allowed, whos_turn

def get_last_move(history: list):

    return None if len(history) == 0 else history[-1] 

def get_sample_card_draw(hole_cards, community_cards = []):
    assert len(hole_cards) + len(community_cards) <= 7


    suits = ['H', 'S', 'C', 'D']
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

    deck = []
    for suit in suits:
        for rank in ranks:
            if suit+rank not in hole_cards + community_cards:
                deck.append(suit+rank)

    shuffle(deck)
    draw_sample = [deck.pop(0) for i in range(9 - len(hole_cards) - len(community_cards))]

    return draw_sample



if __name__ == '__main__':
    from time import perf_counter as pf

    player = MCSPlayer()
    player.imsmallblind = True
    sim_results = player.run_montecarlo_simulation(
        current_street    = 'preflop',
        shist_this_street = ['call'],
        bhist_this_street = ['raise'],
        sbet_so_far       = 20,
        bbet_so_far       = 40,
        hole_cards        = ['H5', 'H3'],
        community_cards   = [],
        sraisecount       = 0,
        braisecount       = 0,
        pot               = 0,
    )
    
    print(sim_results)