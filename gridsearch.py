from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from callplayer import CallPlayer
from mcsplayer import MCSPlayer
from rateplayer import MCSRatePlayer

from time import perf_counter as pf
from math import copysign
from pprint import pprint

#TODO:config the config as our wish
initial_stack = 1000
print('Starting test...')
raise_ratios = [55/40, 60/40, 65/40]
call_ratios  = [20/40, 25/40, 30/40]

wins = dict()
for raise_ratio in raise_ratios:
    for call_ratio in call_ratios:
        payments = []
        for i in range(20):
            t = pf()
            config = setup_config(max_round=500, initial_stack=initial_stack, small_blind_amount=10)


            config.register_player(name=f"mcsmass:{raise_ratio}|{call_ratio}", algorithm=MCSPlayer(call_ratio=call_ratio, raise_ratio=raise_ratio))
            config.register_player(name="raise", algorithm=RaisedPlayer())


            # timer = pf()
            game_result = start_poker(config, verbose=0)
            # timer = pf() - timer
            name = game_result['players'][0]['name']
            stack = game_result['players'][0]['stack']
            payments.append(stack)
            t = pf() - t
            print(t, stack, name)
    

# print (game_result)
# print(f'took {timer} seconds')
        print(sum([(x - initial_stack) for x in payments]))
        print(sum([copysign(1, (x - initial_stack)) for x in payments]))
        print()
        # wins.append(sum([copysign(1, (x - initial_stack)) for x in payments]))
        wins[(raise_ratio, call_ratio)] = sum([copysign(1, (x - initial_stack)) for x in payments])

pprint(wins)

k = list(wins.keys())
v = list(wins.values())
m = max(v)
best_ratios = k[v.index(m)]

print(f'best_ratios: {best_ratios}, value: {wins.get(best_ratios)}, m: {m}')

