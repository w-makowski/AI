import time, copy, random
from typing import Tuple, Optional
from heuristics import *


class MinimaxPlayer:
    def __init__(self, player: str, max_depth: int, heuristic: Heuristic = None, use_alpha_beta: bool = False):
        """
        Initialization of Minimax agent
        :param player: player symbol ('B' or 'W')
        :param max_depth: maximum search depth
        :param heuristic: heuristic class to evaluate game state
        :param use_alpha_beta: whether to use alpha-beta cuts
        """
        self.player = player
        self.opponent = 'W' if player == 'B' else 'B'
        self.max_depth = max_depth
        self.heuristic = heuristic
        self.use_alpha_beta = use_alpha_beta
        self.nodes_visited = 0
        self.execution_time = 0

    def choose_move(self, game: ClobberGame) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Chooses best move for player
        :param game: current game state
        :return: best possible move or None
        """
        start_time = time.time()
        self.nodes_visited = 0

        valid_moves = game.get_valid_moves(self.player)
        if not valid_moves:
            return None

        best_move = None
        best_value = float('-inf')

        for move in valid_moves:
            game_copy = copy.deepcopy(game)
            game_copy.make_move(move)
            self.nodes_visited += 1

            if self.use_alpha_beta:
                value = self.minimax_alpha_beta(game_copy, self.max_depth - 1, float('-inf'), float('inf'), False)
            else:
                value = self.minimax(game_copy, self.max_depth - 1, False)

            if value > best_value:
                best_value = value
                best_move = move

        self.execution_time = time.time() - start_time
        return best_move

    def minimax(self, game: ClobberGame, depth: int, is_maximizing: bool) -> int:
        """
        Minimax algorithm for determining move values
        :param game: game state
        :param depth: remaining search depth
        :param is_maximizing: is this a maximizing level
        :return: best move value
        """

        current_player = self.player if is_maximizing else self.opponent

        # Final Conditions
        if depth == 0 or game.is_game_over(current_player):
            return self.heuristic.evaluate(game, self.player)

        valid_moves = game.get_valid_moves(current_player)

        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(move)
                self.nodes_visited += 1

                t_eval = self.minimax(game_copy, depth - 1, False)
                max_eval = max(max_eval, t_eval)

            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(move)
                self.nodes_visited += 1

                t_eval = self.minimax(game_copy, depth - 1, True)
                min_eval = min(min_eval, t_eval)

            return min_eval

    def minimax_alpha_beta(self, game: ClobberGame, depth: int, alpha: float, beta: float, is_maximizing: bool) -> int:
        """
        Minimax algorithm with alpha-beta cuts
        :param game: game state
        :param depth: remaining search depth
        :param alpha: alpha value
        :param beta: beta value
        :param is_maximizing: is this a maximizing level
        :return: best move value
        """

        current_player = self.player if is_maximizing else self.opponent

        # Final conditions
        if depth == 0 or game.is_game_over(current_player):
            return self.heuristic.evaluate(game, self.player)

        valid_moves = game.get_valid_moves(current_player)

        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(move)
                self.nodes_visited += 1

                t_eval = self.minimax_alpha_beta(game_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, t_eval)

                alpha = max(alpha, t_eval)
                if beta <= alpha:
                    break   # beta cut-off

            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(move)
                self.nodes_visited += 1

                t_eval = self.minimax_alpha_beta(game_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, t_eval)

                beta = min(beta, t_eval)
                if beta <= alpha:
                    break   # alpha cut-off

            return min_eval

