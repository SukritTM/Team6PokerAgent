from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from callplayer import CallPlayer
from mcsplayer import MCSPlayer

from time import perf_counter as pf
from math import copysign

#TODO:config the config as our wish

print('Starting test...')
payments = []
for i in range(10):
    t = pf()
    config = setup_config(max_round=500, initial_stack=10000, small_blind_amount=10)


    config.register_player(name="mcs", algorithm=MCSPlayer())
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
print(sum([(x - 10000) for x in payments]))
print(sum([copysign(1, (x - 10000)) for x in payments]))
