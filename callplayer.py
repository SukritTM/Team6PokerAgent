from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint

class CallPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    print(valid_actions)
    for i in valid_actions:
        if i["action"] == "call":
            action = i["action"]
            return action  # action returned here is sent to the poker engine
    action = valid_actions[1]["action"]
    return action # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

def setup_ai():
  return CallPlayer()
