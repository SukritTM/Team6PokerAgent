from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from callplayer import CallPlayer
from mcsplayer import MCSPlayer

from time import perf_counter as pf

#TODO:config the config as our wish
config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=10)


config.register_player(name="mcs", algorithm=MCSPlayer())
config.register_player(name="raise", algorithm=RaisedPlayer())


timer = pf()
game_result = start_poker(config, verbose=0)
timer = pf() - timer

print (game_result)
print(f'took {timer} seconds')
