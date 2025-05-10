from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint
from stable_baselines3 import PPO
import numpy as np

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['S', 'H', 'D', 'C']

class PPOPlayer(BasePokerPlayer):
  def __init__(self):
    super().__init__()
    self.model = PPO.load("models/ppo_poker_200000_steps.zip", device="cpu")
    self.initial_stack = 1000

  def declare_action(self, valid_actions, hole_card, round_state):
    obs = self._state_to_observation(hole_card, round_state, valid_actions)
    action, state = self.model.predict(obs, deterministic=True)
    action_map = {0: "fold", 1: "call", 2: "raise"}
    poker_action = action_map[int(action)]
    print(f"Poker action: {poker_action}")
    return poker_action
    # pass
  
  # Previous state to observation, unfortunately this one performed the best (still bad lol)
  def _state_to_observation(self, hole_card, round_state, valid_actions):
        def one_hot_52(cards):
            vec = np.zeros(52, dtype=np.float32)
            if cards is None:
                return vec
            for c in cards:
                if not isinstance(c, str) or len(c) != 2:
                    continue
                suit = c[0].upper()
                rank = c[1].upper()
                if rank in RANKS and suit in SUITS:
                    idx = SUITS.index(suit) * 13 + RANKS.index(rank)
                    vec[idx] = 1.0
            return vec
        community = round_state["community_card"]
        next_player_pos = round_state["next_player"]
        
        agent_seat = round_state["seats"][next_player_pos]
        agent_uuid = agent_seat["uuid"]
        
        opponent_seat = next(s for s in round_state["seats"] if s["uuid"] != agent_uuid)
        
        agent_stack = agent_seat["stack"]
        self.curr_stack = agent_stack
        opponent_stack = opponent_seat["stack"]
        
        street = round_state["street"].lower()

        pot_amount = round_state["pot"]["main"]["amount"]
        
        sb_pos = round_state.get("small_blind_pos")
        sb_flag = 1.0 if next_player_pos == sb_pos else 0.0

        street_vec = np.zeros(4, dtype=np.float32)
        street_idx = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(street, 0)
        street_vec[street_idx] = 1.0

        can_raise = 1.0 if any(d['action'] == 'raise' for d in valid_actions) else 0.0
             
    
        obs = np.concatenate([
            one_hot_52(hole_card),
            one_hot_52(community),
            np.array([
                agent_stack / self.initial_stack,
                opponent_stack / self.initial_stack,
                pot_amount / (2 * self.initial_stack),
                sb_flag
            ], dtype=np.float32),
            street_vec,
            np.array([can_raise])
        ])
        
        return obs.astype(np.float32)

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
