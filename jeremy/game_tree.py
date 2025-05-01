import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import csv

from Team6PokerAgent.jeremy.notebook import TypedPokerPlayer, ActionOption, CardString, RoundState, Action
from pypokerengine.api.game import setup_config, start_poker

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GameTreeNode:
    player_to_act: int
    street: str
    actions: Dict[str, "GameTreeNode"]
    pot_size: int
    is_terminal: bool = False
    terminal_type: Optional[str] = None


tree_root = GameTreeNode(player_to_act=0, street="root", actions={}, pot_size=0)

MAX_DEPTH = 10
MAX_SIMULATIONS = 1000
simulations_run = 0

def hash_round_state(round_state):
    return round_state["street"], round_state["pot"]["main"]["amount"]

def print_game_tree(node: GameTreeNode, indent: str = ""):
    term = " (Terminal)" if node.is_terminal else ""
    extra = f" [{node.terminal_type}]" if node.terminal_type else ""
    print(f"{indent}Player {node.player_to_act}, Street: {node.street}, Pot: {node.pot_size}{term}{extra}")
    for action, child in node.actions.items():
        print(f"{indent}  Action: {action}")
        print_game_tree(child, indent + "    ")

def export_to_json(node: GameTreeNode) -> dict:
    return {
        "player_to_act": node.player_to_act,
        "street": node.street,
        "pot_size": node.pot_size,
        "is_terminal": node.is_terminal,
        "terminal_type": node.terminal_type,
        "actions": {k: export_to_json(v) for k, v in node.actions.items()}
    }

def export_to_dot(node: GameTreeNode, path="root", dot=None):
    if dot is None:
        dot = ["digraph GameTree {"]
    label = f"{path}\nP{node.player_to_act} {node.street} {node.pot_size}" + (f"\\n{node.terminal_type}" if node.terminal_type else "")
    dot.append(f'"{path}" [label="{label}"]')
    for action, child in node.actions.items():
        child_path = path + "/" + action
        dot.append(f'"{path}" -> "{child_path}" [label="{action}"]')
        export_to_dot(child, child_path, dot)
    if path == "root":
        dot.append("}")
        return "\n".join(dot)
    return dot

def export_to_csv_rows(node: GameTreeNode, path: List[str], rows: List[List[str]]):
    row = [
        "->".join(path),
        str(node.player_to_act),
        node.street,
        str(node.pot_size),
        str(node.is_terminal),
        node.terminal_type or ""
    ]
    rows.append(row)
    for action, child in node.actions.items():
        export_to_csv_rows(child, path + [action], rows)

class FixedPlayer(TypedPokerPlayer):
    def __init__(self, declare_action_cb: callable):
        super().__init__()
        self.declare_action_cb = declare_action_cb

    def declare_action(self, valid_actions: List[ActionOption], hole_card: List[CardString], round_state: RoundState) -> Action:
        return self.declare_action_cb(valid_actions, hole_card, round_state)

class CallbackContext:
    def __init__(self, root_node: GameTreeNode, path: List[str]):
        self.last_action_taken: Optional[str] = None
        self.root_node = root_node
        self.path_so_far: List[str] = []
        self.full_path: List[str] = path.copy()
        self.new_action: Optional[str] = None
        self.new_node_added: Optional[GameTreeNode] = None
        self.was_cb_called: bool = False
        self.previous_street: Optional[str] = None

    def current_node(self) -> GameTreeNode:
        node = self.root_node
        for action in self.path_so_far:
            node = node.actions[action]
        return node

def explore(path: List[str], node: GameTreeNode):
    if node.is_terminal:
        logger.info(f"Reached terminal node at path: {path}")
        return
    if len(path) >= MAX_DEPTH:
        logger.warning(f"Max depth reached at path: {path}")
        node.is_terminal = True
        return
    logger.info(f"Exploring path: {path}")
    while True:
        new_action = simulate_one_game(path, node)
        if new_action is None:
            if not node.actions or all(child.is_terminal for child in node.actions.values()):
                logger.info(f"Marking node as terminal after exhausting actions: {path}")
                node.is_terminal = True
                if node.terminal_type is None:
                    node.terminal_type = "showdown"
            break
        if new_action in node.actions:
            explore(path + [new_action], node.actions[new_action])
        else:
            logger.warning(f"Action {new_action} returned but not in node.actions at path: {path}")

def simulate_one_game(path: List[str], node: GameTreeNode) -> Optional[str]:
    last_valid_actions: Optional[List[ActionOption]] = None
    global simulations_run
    if simulations_run >= MAX_SIMULATIONS:
        logger.critical("Reached max simulation count.")
        exit(1)

    simulations_run += 1
    logger.debug(f"Simulation #{simulations_run}, Path: {path}")

    ctx = CallbackContext(root_node=node, path=path)

    def cb(valid_actions: List[ActionOption], hole_card: List[CardString], round_state: RoundState) -> Action:
        ctx.was_cb_called = True
        nonlocal last_valid_actions
        last_valid_actions = valid_actions
        street, pot = hash_round_state(round_state)
        current = ctx.current_node()

        if current.street == "root":
            current.street = street
            current.pot_size = pot

        if current.street != street:
            current.street = street
            current.pot_size = pot

        ctx.previous_street = street

        if not valid_actions:
            logger.info(f"Marking terminal node due to no actions at street {street}, pot {pot}, path: {ctx.full_path + ctx.path_so_far}")
            current.is_terminal = True
            current.terminal_type = "no_actions"
            return "fold"

        unexplored = [a["action"] for a in valid_actions if a["action"] not in current.actions]

        if not unexplored:
            action = valid_actions[0]["action"]
        else:
            PREF_ORDER = ["raise", "call", "fold"]
            action = next(a for a in PREF_ORDER if a in unexplored)
            logger.info(f"Adding new node for unexplored action: {action} at path: {ctx.full_path + ctx.path_so_far}")
            is_terminal = action == "fold"
            terminal_type = "fold" if is_terminal else None
            next_node = GameTreeNode(
                player_to_act=(current.player_to_act + 1) % 2,
                street=street,
                actions={},
                pot_size=pot,
                is_terminal=is_terminal,
                terminal_type=terminal_type
            )
            current.actions[action] = next_node
            ctx.new_action = action
            ctx.new_node_added = next_node

        ctx.last_action_taken = action
        ctx.path_so_far.append(action)

        return action

    game_finished = False

    class EndingFixedPlayer(FixedPlayer):
        def declare_action(self, valid_actions, hole_card, round_state):
            nonlocal game_finished
            if not valid_actions:
                game_finished = True
            return super().declare_action(valid_actions, hole_card, round_state)

    p1 = EndingFixedPlayer(declare_action_cb=cb)
    p2 = EndingFixedPlayer(declare_action_cb=cb)
    config = setup_config(max_round=1, initial_stack=10000, small_blind_amount=10)
    config.register_player(name="p1", algorithm=p1)
    config.register_player(name="p2", algorithm=p2)
    start_poker(config, verbose=0)

    valid_action_names = [a["action"] for a in last_valid_actions] if last_valid_actions else []
    if all(a in node.actions and node.actions[a].is_terminal for a in valid_action_names):
        logger.info(f"All valid actions exhausted — marking node as terminal: {path}")
        node.is_terminal = True
        if node.terminal_type is None:
            node.terminal_type = "showdown"
    else:
        logger.debug(f"Simulation ended with new action: {ctx.new_action}")

    if not ctx.was_cb_called or game_finished or (last_valid_actions is None or not last_valid_actions):
        logger.info(f"Simulation finished — marking node terminal at path: {path}")
        node.is_terminal = True
        if node.terminal_type is None:
            node.terminal_type = "showdown"
        return None
    return ctx.new_action if ctx.new_node_added is not None else None

def main():
    explore([], tree_root)
    print("\nFinal Game Tree:")
    print_game_tree(tree_root)

    # Export JSON
    with open("../../project/AI-Poker-Agent/game_tree.json", "w") as f:
        json.dump(export_to_json(tree_root), f, indent=2)

    # Export DOT
    with open("../../project/AI-Poker-Agent/game_tree.dot", "w") as f:
        f.write(export_to_dot(tree_root))

    # Export CSV
    rows = [["Path", "PlayerToAct", "Street", "PotSize", "IsTerminal", "TerminalType"]]
    export_to_csv_rows(tree_root, [], rows)
    with open("../../project/AI-Poker-Agent/game_tree.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == "__main__":
    main()
