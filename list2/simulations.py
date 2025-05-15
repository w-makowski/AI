import csv
import os


class Simulator:
    def run_simulation(self, game, heuristics):

        game = game
        heuristics = heuristics

        print(
            'test1 - Agent 1 (Edge Heuristic, max depth: 3, alpha - beta: true) Agent 2 (Mobile Piece Count Heuristic, max depth: 3, alpha - beta: true)')
        game.play_game(player1_heuristic=heuristics['EdgeHeuristic'], player2_heuristic=heuristics['MobilePieceCountHeuristic'])
        print()
        print(
            'test2 - Agent 1 (Edge Heuristic, max depth: 4, alpha - beta: true) Agent 2 (Mobile Piece Count Heuristic, max depth: 4, alpha - beta: true)')
        game.play_game(player1_heuristic=heuristics['EdgeHeuristic'], player2_heuristic=heuristics['MobilePieceCountHeuristic'], max_depth=4)
        print()
        print(
            'test3 - Agent 1 (Edge Heuristic, max depth: 3, alpha - beta: false) Agent 2 (Mobile Piece Count Heuristic, max depth: 3, alpha - beta: false)')
        game.play_game(player1_heuristic=heuristics['EdgeHeuristic'], player2_heuristic=heuristics['MobilePieceCountHeuristic'], use_alpha_beta=False)
        print()
        print(
            'test4 - Agent 1 (Edge Heuristic, max depth: 4, alpha - beta: false) Agent 2 (Mobile Piece Count Heuristic, max depth: 4, alpha - beta: false)')
        game.play_game(player1_heuristic=heuristics['EdgeHeuristic'], player2_heuristic=heuristics['MobilePieceCountHeuristic'], use_alpha_beta=False, max_depth=4)
        print()
        print(
            'test5 - Agent 1 (Edge Heuristic, max depth: 5, alpha - beta: true) Agent 2 (Mobile Piece Count Heuristic, max depth: 5, alpha - beta: true)')
        game.play_game(player1_heuristic=heuristics['EdgeHeuristic'],
                       player2_heuristic=heuristics['MobilePieceCountHeuristic'], max_depth=5)
        print()
        print(
            'test6 - Agent 1 (Edge Heuristic, max depth: 6, alpha - beta: true) Agent 2 (Mobile Piece Count Heuristic, max depth: 6, alpha - beta: true)')
        game.play_game(player1_heuristic=heuristics['EdgeHeuristic'],
                       player2_heuristic=heuristics['MobilePieceCountHeuristic'], max_depth=6)
        print()

    def run_tournament(self, game, heuristics):
        test_counter = 0
        csv_file = 'tournament_depth4.csv'
        txt_file = 'tournament_depth4.txt'
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as csvf, open(txt_file, 'a') as txtf:
            fieldnames = ['test_id', 'B heuristic', 'W heuristic', 'winner', 'rounds', 'nodes_white', 'nodes_black',
                          'total_nodes', 'time_white', 'time_black', 'total_time']
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for player_h in heuristics:
                for opponent_h in heuristics:
                    header = f'Test{test_counter} {player_h} vs {opponent_h}:'
                    print(header)
                    stats = game.play_game(
                        player1_heuristic=heuristics[player_h],
                        player2_heuristic=heuristics[opponent_h],
                        max_depth=4
                    )

                    txtf.write(header + '\n')
                    for key, value in stats.items():
                        txtf.write(f'{key}: {value}\n')
                    txtf.write('\n')

                    row = {'test_id': test_counter, 'B heuristic': player_h, 'W heuristic': opponent_h}
                    row.update(stats)
                    writer.writerow(row)

                    test_counter += 1


