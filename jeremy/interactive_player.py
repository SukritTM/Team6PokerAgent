from typing import List

from notebook import TypedPokerPlayer, ActionOption, CardString, RoundState, Action
from pypokerengine.api.game import setup_config, start_poker
from raise_player import RaisedPlayer
from pprint import pprint
from colorama import Fore, Style

class InteractivePlayer(TypedPokerPlayer):
    def __init__(self, color: str = Fore.RED):
        super().__init__()
        self.color = color

    def declare_action(self, valid_actions: List[ActionOption], hole_card: List[CardString], round_state: RoundState) -> Action:
        print(f"\n{self.color}Round State:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}", end="")  # Set color
        pprint(round_state)
        print(f"{Style.RESET_ALL}", end="")  # Reset color
        print(f"\n{self.color}Hole Cards:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.GREEN}", end="")
        pprint(hole_card)
        print(f"{Style.RESET_ALL}", end="")  # Reset color
        print(f"\n{self.color}Valid Actions:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.YELLOW}", end="")
        for i, action in enumerate(valid_actions):
            print(f"{i}: ", end="")
            pprint(action)
        print(f"{Style.RESET_ALL}", end="")  # Reset color

        choice = input("Choose an action (number): ")
        while not choice.isdigit() or int(choice) >= len(valid_actions):
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
            choice = input("Choose an action (number): ")
        return valid_actions[int(choice)]["action"]

    # Other methods remain unchanged

    def receive_game_start_message(self, game_info: dict) -> None:
        print(f"\n{self.color}Game started with info:{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}", end="")
        pprint(game_info)
        print(f"{Style.RESET_ALL}", end="")  # Reset color

    def receive_round_start_message(self, round_count: int, hole_card: List[CardString], seats: List[dict]) -> None:
        print(f"\n{self.color}Round {round_count} started with hole cards:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}", end="")
        pprint(hole_card)
        print(f"{Style.RESET_ALL}", end="")  # Reset color
        print(f"\n{self.color}Seats:{Style.RESET_ALL}")
        for seat in seats:
            if seat['uuid'] == self.uuid:
                print(f"{self.color}", end="")
                pprint(seat)
                print(f"{Style.RESET_ALL}", end="")
            else:
                print(f"{Fore.WHITE}", end="")
                pprint(seat)
                print(f"{Style.RESET_ALL}", end="")

    def receive_street_start_message(self, street: str, round_state: RoundState) -> None:
        print(f"\n{self.color}Street started:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.CYAN}{street}{Style.RESET_ALL}", end=" ")

        print(f"\n{self.color}Round State:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.YELLOW}", end="")
        pprint(round_state)
        print(f"{Style.RESET_ALL}", end="")

    def receive_game_update_message(self, action: dict, round_state: RoundState) -> None:
        print(f"\n{self.color}Game updated with action:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.MAGENTA}", end="")
        pprint(action)
        print(f"{Style.RESET_ALL}", end="")
        print(f"\n{self.color}Round State:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.YELLOW}", end="")
        pprint(round_state)

    def receive_round_result_message(self, winners: List[dict], hand_info: List[dict], round_state: RoundState) -> None:
        print(f"\n{self.color}Round result received:{Style.RESET_ALL}", end=" ")
        print(f"\n{self.color}Winners:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.GREEN}", end="")
        pprint(winners)
        print(f"{Style.RESET_ALL}", end="")
        print(f"\n{self.color}Hand Info:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.YELLOW}", end="")
        pprint(hand_info)
        print(f"{Style.RESET_ALL}", end="")
        print(f"\n{self.color}Round State:{Style.RESET_ALL}", end=" ")
        print(f"{Fore.CYAN}", end="")
        pprint(round_state)
        print(f"{Style.RESET_ALL}", end="")



def main():
    # Example usage
    print("Starting interactive poker game...")
    p1 = InteractivePlayer(color=Fore.RED)
    p2 = InteractivePlayer(color=Fore.BLUE)
    config = setup_config(max_round=1, initial_stack=10000, small_blind_amount=10)
    config.register_player(name="p1", algorithm=p1)
    config.register_player(name="p2", algorithm=p2)
    game_result = start_poker(config, verbose=1)

if __name__ == "__main__":
    main()