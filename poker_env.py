import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from pypokerengine.api.emulator import Emulator
import numpy as np
from randomplayer import RandomPlayer
from PPOplayer import PPOPlayer

register(
    id='pokerEnv',
    entry_point='poker_env:PokerEnv',
)

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'] # The card ranks
SUITS = ['S', 'H', 'D', 'C'] # The suits -- spades, hearts, diamonds, clubs

class PokerEnv(gym.Env):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, agent_model = PPOPlayer(), opponent_model = RandomPlayer(), player_num = 2, max_round = 500, initial_stack = 1000, small_blind_amount = 10, ante_amount = 0):
        super().__init__()

        self.emulator = Emulator()
        self.player_num = player_num
        self.max_round = max_round
        self.initial_stack = initial_stack
        self.small_blind_amount = small_blind_amount
        self.ante_amount = ante_amount

        # Initialize poker game:
        self.emulator.set_game_rule(self.player_num, self.max_round, self.small_blind_amount, self.ante_amount)
        
        # Fold, Call, Raise for the action space
        self.action_space = spaces.Discrete(3)
        self.valid_actions = []


        # Initialize the observation space:
        # The shape of the observation space is of length 114 in the following fashion:

        # 0-51 are for your hand cards -- so two will be 1's and the rest will be 0's
        # 52-103 are for the board cards -- after the river, there will be 5 1's and the rest 0's
        # 104 is for the agent's stack and will be a fraction based on the initial stack
        # 105 is for the opponent's stack and will be a fraction based on the initial stack
        # 106 is for the pot and will be a fraction based on 2x the initial stack size
        # 107 is whether or not the agent is small blind (1) or big blind (0)
        # 108-111 is for the street -- pre-flop, flop, turn, river
        # 112 is an indicator for whether or not raising is a valid move
        # 113 is whether or not it costs money to call or not

        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (114,),
            dtype = np.float32
        )

        self.previous_observation = None

        self.curr_stack = initial_stack
        self.stack_at_hand_start = initial_stack

        self.agent_model = agent_model
        self.opponent_model = opponent_model

        self.agent_uuid = "agent-uuid"
        self.opponent_uuid = "opponent-uuid"
        
        self.emulator.register_player(self.agent_uuid, self.agent_model)
        self.emulator.register_player(self.opponent_uuid, self.opponent_model)

        self.players_info = {
            self.agent_uuid : {"stack": self.initial_stack, "name": "agent"},
            self.opponent_uuid : {"stack": self.initial_stack, "name": "opponent"},
        }

        # Create an empty initial game state given the players:
        self.state = self.emulator.generate_initial_game_state(self.players_info)


    # This is in order to convert the state of the game to the observation vector that we are expecting
    def _state_to_observation(self, hole_card, round_state, valid_actions):
        # helper function to convert cards to one-hot encoding
        def one_hot_52(cards):
            vec = np.zeros(52, dtype=np.float32)
            for c in cards:
                if not isinstance(c, str) or len(c) != 2:
                    continue  # skip cards with issues (hopefulyl should be none)
                suit = c[0].upper()
                rank = c[1].upper()
                if rank in RANKS and suit in SUITS:
                    idx = SUITS.index(suit) * 13 + RANKS.index(rank)
                    vec[idx] = 1.0
            return vec

        # Get the game state information
        community = round_state["community_card"]
        
        agent_pos = next(i for i,s in enumerate(round_state["seats"]) if s["uuid"] == self.agent_uuid)
        agent_seat = round_state["seats"][agent_pos]
        opponent_seat = round_state["seats"][1 - agent_pos]
        
        # Get the size of each players stack
        agent_stack = agent_seat["stack"]
        self.curr_stack = agent_stack
        opponent_stack = opponent_seat["stack"]
    
        street = round_state["street"].lower()
        
        # Get the pot size, assuming that there is no side 
        pot_amount = round_state["pot"]["main"]["amount"]
        
        # Check to see if current player is small blind
        sb_pos  = round_state.get("small_blind_pos")
        sb_flag = 1.0 if agent_pos == sb_pos else 0.0
        
        # Encode the street to one-hot
        street_vec = np.zeros(4, dtype=np.float32)
        street_idx = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(street, 0)
        street_vec[street_idx] = 1.0
        
        # Helper function to see if in order to call, it costs money
        def to_call_flag(agent_uuid: str, rs: dict) -> float:
            street = rs["street"].lower()
            hist = rs["action_histories"][street]
            current_bet = max((h["amount"] for h in hist), default=0)
            your_bet = max((h["amount"] for h in hist if h["uuid"] == agent_uuid), default=0)
            return 1.0 if current_bet > your_bet else 0.0

        call_cost_flag = to_call_flag(self.agent_uuid, round_state)

        can_raise = 1.0 if any(d['action'] == 'raise' for d in valid_actions) else 0.0

        obs = np.concatenate([
            one_hot_52(hole_card), # 0-51
            one_hot_52(community), # 52-103
            np.array([
                agent_stack   / self.initial_stack, # 104
                opponent_stack/ self.initial_stack, # 105
                pot_amount    / (2*self.initial_stack), # 106
                sb_flag # 107
            ], dtype=np.float32),
            street_vec, # 108-111
            np.array([
                can_raise, # 112
                call_cost_flag # 113 
            ], dtype=np.float32)
        ])
        
        return obs.astype(np.float32)
    
    def _make_opponent_move(self, state, round_state, valid_actions, hole_cards):
        # print("ENV: Making Opponent Move")
        action = self.opponent_model.declare_action(valid_actions, hole_cards, round_state)
        new_state, events = self.emulator.apply_action(state, action)
        return new_state, events

    def _reset_round(self):
        # print("ENV: Resetting the round")
        self.stack_at_hand_start = self.curr_stack
        self.state, events = self.emulator.start_new_round(self.state)
        
        round_state = None
        curr_player_uuid = None
        valid_actions = None
        hole_cards = None
        terminated = False
        
        for e in events:
            if e["type"] == "event_ask_player":
                round_state = e["round_state"]
                curr_player_uuid = e["uuid"]
                valid_actions = e["valid_actions"]
                self.valid_actions = valid_actions
                hole_cards = self.emulator.get_hole_cards(self.state)
            elif e["type"] == "event_game_finish":
                # print("GAME FINISH in reset_round")
                terminated = True
                obs = self.previous_observation if hasattr(self, 'previous_observation') else np.zeros(self.observation_space.shape)
                return obs, terminated
        
        if curr_player_uuid == self.opponent_uuid:
            self.state, events = self._make_opponent_move(self.state, round_state, valid_actions, hole_cards)
            for e in events:
                if e["type"] == "event_game_finish":
                    # print("GAME FINISH in reset_round")
                    terminated = True
                    obs = self.previous_observation if hasattr(self, 'previous_observation') else np.zeros(self.observation_space.shape)
                    return obs, terminated
            for e in events:
                if e["type"] == "event_ask_player":
                    round_state = e["round_state"]
                    curr_player_uuid = e["uuid"]
                    valid_actions = e["valid_actions"]
                    self.valid_actions = valid_actions
                    hole_cards = self.emulator.get_hole_cards(self.state)
        
        obs = self._state_to_observation(hole_cards, round_state, valid_actions)
        self.previous_observation = obs
    
        return obs, terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # print("ENV: Resetting the game")
        self.emulator = Emulator()
        
        self.emulator.set_game_rule(self.player_num, self.max_round, self.small_blind_amount, self.ante_amount)

        self.emulator.register_player(self.agent_uuid, self.agent_model)
        self.emulator.register_player(self.opponent_uuid, self.opponent_model)

        self.players_info = {
            self.agent_uuid : {"stack": self.initial_stack, "name": "agent"},
            self.opponent_uuid : {"stack": self.initial_stack, "name": "opponent"},
        }

        self.state = self.emulator.generate_initial_game_state(self.players_info)
        
        obs, terminated = self._reset_round()
        if terminated == True:
            raise Exception("Something went very very wrong I hope this never shows up lol")
            
        return obs, {}
    
    # This didn't end up working unfortunately
    def _calculate_reward(self, event, previous_stack=None, chosen_action=None):

        SB = self.small_blind_amount

        if event is not None:
            etype = event["type"]

            if etype == "event_game_finish":
                final_stack = next(p["stack"] for p in event["players"] if p["uuid"] == self.agent_uuid)
                return final_stack - self.initial_stack

            if etype == "event_round_finish":
                seats = event["round_state"]["seats"]
                round_stack = next(p["stack"] for p in seats if p["uuid"] == self.agent_uuid)

                base = round_stack - self.stack_at_hand_start
                street_finished = event["round_state"]["street"].lower()

                if street_finished == "preflop" and chosen_action == "fold":
                    base -= SB

                if street_finished == "river" and base > 0:
                    base += 2 * SB

                return base

            return 0.0

        if previous_stack is None:
            return 0.0

        step_delta = self.curr_stack - previous_stack 

        if chosen_action == "fold" and self.previous_observation is not None:
            call_cost_flag = self.previous_observation[113]
            if call_cost_flag == 0:
                step_delta -= 0.25 * SB

        return step_delta
        # return 0.0


    def step(self, action):
        action_map = {0: "fold", 1: "call", 2: "raise"}
        poker_action = action_map[action]

        terminated, truncated = False, False
        prev_stack = self.curr_stack

        if poker_action not in [va["action"] for va in self.valid_actions]:
            return (self.previous_observation, -1000, True, truncated, {"error": "Invalid action"})

        self.state, events = self.emulator.apply_action(self.state, poker_action)

        round_state = valid_actions = hole_cards = None

        for e in events:
            t = e["type"]
            if t == "event_game_finish":
                return (self.previous_observation, self._calculate_reward(e), True, truncated, {"Event": "Game Ended"})

            if t == "event_round_finish":
                if e["round_state"]["round_count"] >= self.max_round-1:
                    return (self.previous_observation, self._calculate_reward(e), True, truncated, {"Event": "Game Ended"})
                obs, terminated = self._reset_round()
                return (obs, self._calculate_reward(e), terminated, truncated, {"Event": "Round Ended"})

            if t == "event_ask_player":
                round_state   = e["round_state"]
                valid_actions = e["valid_actions"]
                self.valid_actions = valid_actions
                hole_cards    = self.emulator.get_hole_cards(self.state)
                break

        self.state, events = self._make_opponent_move(self.state, round_state, valid_actions, hole_cards)

        for e in events:
            t = e["type"]
            if t == "event_game_finish":
                return (self.previous_observation, self._calculate_reward(e, chosen_action=poker_action), True, truncated, {"Event": "Game Ended"})

            if t == "event_round_finish":
                if e["round_state"]["round_count"] >= self.max_round-1:
                    return (self.previous_observation, self._calculate_reward(e, chosen_action=poker_action), True, truncated, {"Event": "Game Ended"})
                obs, terminated = self._reset_round()
                return (obs, self._calculate_reward(e), terminated, truncated, {"Event": "Round Ended"})

            if t == "event_ask_player":
                round_state   = e["round_state"]
                valid_actions = e["valid_actions"]
                self.valid_actions = valid_actions
                hole_cards = self.emulator.get_hole_cards(self.state)
                break

        obs = self._state_to_observation(hole_cards, round_state, valid_actions)
        self.previous_observation = obs
        reward = self._calculate_reward(None, chosen_action=poker_action, previous_stack=prev_stack)

        return obs, reward, terminated, truncated, {"Event": "Step taken"}
