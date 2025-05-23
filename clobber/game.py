from minimax import MinimaxPlayer
from heuristics import *
from simulations import Simulator


class Game:

    def play_game(self,
                  rows: int = 5,
                  cols: int = 6,
                  max_depth: int = 3,
                  player1_heuristic: Heuristic = None,
                  player2_heuristic: Heuristic = None,
                  use_alpha_beta: bool = True,
                  show_every_round: bool = False,
                  load_board: bool = False):
        """
        Plays a full Clobber game
        :param rows: number of board rows
        :param cols: number of board columns
        :param max_depth: maximum search depth
        :param player1_heuristic: heuristic for player 1 (black)
        :param player2_heuristic: heuristic for player 2 (white)
        :param use_alpha_beta: whether to use alpha-beta cuts
        :param show_every_round: prints every round (player that made move, move, board state after move)
        :param load_board: whether to load board from input
        :return: stats: dictionary with stats from the game
        """
        game = ClobberGame(rows, cols)
        stats = {'winner': '_', 'rounds': 0, 'nodes_white': 0, 'nodes_black': 0, 'total_nodes': 0,
                 'time_white': 0, 'time_black': 0, 'total_time': 0}

        if load_board:
            game.load_board_from_input()

        player1 = MinimaxPlayer('B', max_depth, player1_heuristic, use_alpha_beta)
        player2 = MinimaxPlayer('W', max_depth, player2_heuristic, use_alpha_beta)

        current_player = player1
        round_count = 0

        if show_every_round:
            print("Initial board state:")
            game.print_board()
            print()

        total_nodes_visited = 0
        total_execution_time = 0

        while True:
            round_count += 1
            if show_every_round:
                print(f"Round {round_count}, player: {current_player.player}")

            move = current_player.choose_move(game)
            total_nodes_visited += current_player.nodes_visited
            total_execution_time += current_player.execution_time

            if current_player.player == 'W':
                stats['time_white'] += current_player.execution_time
                stats['nodes_white'] += current_player.nodes_visited
            else:
                stats['time_black'] += current_player.execution_time
                stats['nodes_black'] += current_player.nodes_visited

            if move is None:
                print(f"Player {current_player.player} can't make a move.")
                print(f"Player {'W' if current_player.player == 'B' else 'B'} wins!")
                stats['winner'] = 'W' if current_player.player == 'B' else 'B'
                break

            if show_every_round:
                print(f"Move: {move[0]} -> {move[1]}")
            game.make_move(move)

            if show_every_round:
                print("Board state after move:")
                game.print_board()
                print()

            # Change player
            current_player = player2 if current_player == player1 else player1

        print("\nFinal state of the board:")
        game.print_board()
        stats['rounds'] = round_count
        print(f"{round_count} {('W' if current_player.player == 'B' else 'B')}")

        # print(f"\nStats:", file=sys.stderr)
        # print(f"Total nodes visited: {total_nodes_visited}", file=sys.stderr)
        # print(f"Total execution time: {total_execution_time:.4f} s", file=sys.stderr)
        print(f"\nStats:")
        print(f"Total nodes visited: {total_nodes_visited}",)
        print(f"Total execution time: {total_execution_time:.4f} s")
        stats['total_nodes'] = total_nodes_visited
        stats['total_time'] = total_execution_time
        print(f"Total nodes visited by W: {stats['nodes_white']}")
        print(f"Total execution time by W: {stats['time_white']}")
        print(f"Total nodes visited by B: {stats['nodes_black']}")
        print(f"Total execution time by B: {stats['time_black']}")

        return stats


def create_custom_game_with_cli():
    rows = int(input("Enter rows number (default = 5): ") or 5)
    cols = int(input("Enter columns number (default = 6): ") or 6)
    max_depth = int(input("Enter max search depth (default=3): ") or 3)

    print("\nAvailable heuristics:")
    print("1. Mobility (number of possible moves)")
    print("2. Number of pieces")
    print("3. Position (board edge preference)")
    print("4. Clustering (cluster together pieces preference)")
    print("5. Combination 1: 3*mobility + 1*number of pieces")
    print("6. Combination 2: 2*mobility + 2*position + 1*cluster")
    print("7. Combination 3: 1*mobility + 1*number of pieces + 3*cluster")

    heuristics = {
        'MobilityHeuristic': MobilityHeuristic(),
        'MobilePieceCountHeuristic': MobilePieceCountHeuristic(),
        'EdgeHeuristic': EdgeHeuristic(),
        'ImprovedPositionHeuristic': ImprovedPositionHeuristic(),
        'IsolationHeuristic': IsolationHeuristic(),
        'EndgameHeuristic': EndgameHeuristic(),
        'ControlZoneHeuristic': ControlZoneHeuristic(),
        'ClusteringHeuristic': ClusteringHeuristic(),
        'SurvivabilityHeuristic': SurvivabilityHeuristic(),
        'CentralControlHeuristic': CentralControlHeuristic(),
        'TrapHeuristic': TrapHeuristic(),
        'OptimizedCombinedHeuristic': OptimizedCombinedHeuristic(),
        'StageBasedHeuristic': StageBasedHeuristic(),
        'StyleMatchupHeuristic': StyleMatchupHeuristic()
    }

    player1_heuristic_idx = int(input("\nSelect heuristic for player 1 (black) [1-7]: ")) - 1
    player2_heuristic_idx = int(input("Select heuristic for player 2 (white) [1-7]: ")) - 1

    player1_heuristic = heuristics[player1_heuristic_idx]
    player2_heuristic = heuristics[player2_heuristic_idx]

    use_alpha_beta = input("\nUse alpha-beta cuts? (y/n, default y): ").lower() != 'n'
    show_every_round = input("\nPrint board state after every move? (y/n, default n): ").lower() != 'n'

    game = Game()
    game.play_game(rows, cols, max_depth, player1_heuristic, player2_heuristic, use_alpha_beta, show_every_round=show_every_round, load_board=True)


def setup_for_simulations():
    heuristics = {
        'MobilityHeuristic': MobilityHeuristic(),
        'MobilePieceCountHeuristic': MobilePieceCountHeuristic(),
        'EdgeHeuristic': EdgeHeuristic(),
        'ImprovedPositionHeuristic': ImprovedPositionHeuristic(),
        'IsolationHeuristic': IsolationHeuristic(),
        'EndgameHeuristic': EndgameHeuristic(),
        'ControlZoneHeuristic': ControlZoneHeuristic(),
        'ClusteringHeuristic': ClusteringHeuristic(),
        'SurvivabilityHeuristic': SurvivabilityHeuristic(),
        'CentralControlHeuristic': CentralControlHeuristic(),
        'TrapHeuristic': TrapHeuristic(),
        'OptimizedCombinedHeuristic': OptimizedCombinedHeuristic(),
        'StageBasedHeuristic': StageBasedHeuristic(),
        'StyleMatchupHeuristic': StyleMatchupHeuristic()
    }

    game = Game()

    simulator = Simulator()

    return game, simulator, heuristics


def run_simulations(game, simulator, heuristics):
    simulator.run_simulation(game, heuristics)


def run_tournament(game, simulator, heuristics):
    simulator.run_tournament(game,heuristics)


def main():
    print("Welcome to Clobber game with Minimax algorithm!")
    print("Options:")
    print("1. Run simulations")
    print("2. Run tournament")
    print("3. Create custom game with CLI")

    user_selection = int(input("Select available option: ") or 1)

    game, simulator, heuristics = setup_for_simulations()

    if user_selection == 1:
        run_simulations(game, simulator, heuristics)
    elif user_selection == 2:
        run_tournament(game, simulator, heuristics)
    else:
        create_custom_game_with_cli()


if __name__ == '__main__':
    main()
