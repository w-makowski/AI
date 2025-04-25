from abc import ABC, abstractmethod
from clobber import ClobberGame


class Heuristic(ABC):
    """
    Interface
    """

    @abstractmethod
    def evaluate(self, game: ClobberGame, player: str) -> int:
        """
        Evaluates heuristic
        :param game: game state
        :param player: player ('B' or 'W')
        :return: heuristic value
        """
        pass


class MobilityHeuristic(Heuristic):
    """
    Mobility heuristic - evaluates the number of possible player moves, the more, the better
    """
    def evaluate(self, game: ClobberGame, player: str) -> int:
        """
        Evaluates the number of possible player moves, the more, the better
        :param game: game state
        :param player: player ('B' or 'W')
        :return: The difference between the number of player moves and the number of opponent moves
        """
        opponent = 'W' if player == 'B' else 'B'
        player_moves = len(game.get_valid_moves(player))
        opponent_moves = len(game.get_valid_moves(opponent))
        return player_moves - opponent_moves


class MobilePieceCountHeuristic(Heuristic):
    """
    Mobile Piece Count Heuristic - evaluates the difference in the number of pieces that can be moved
    """
    def evaluate(self, game: ClobberGame, player: str) -> int:
        """
        Evaluates the difference in the number of pieces that can be moved
        :param game: game state
        :param player: player ('B' or 'W')
        :return: The difference between the player's number of pieces and the opponent's number of pieces
        """
        opponent = 'W' if player == 'B' else 'B'
        player_mobile_pieces = set()
        opponent_mobile_pieces = set()
        for move in game.get_valid_moves(player):
            (x0, y0), _ = move
            player_mobile_pieces.add((x0, y0))
        for move in game.get_valid_moves(opponent):
            (x0, y0), _ = move
            opponent_mobile_pieces.add((x0, y0))
        return len(player_mobile_pieces) - len(opponent_mobile_pieces)


class PositionHeuristic(Heuristic):
    """
    Position Heuristic - prefers pieces (player's figures) on the edges of the board
    """
    def evaluate(self, game: ClobberGame, player: str) -> int:
        """
        Evaluates heuristic
        :param game: game state
        :param player: player ('B' or 'W')
        :return: The difference between player position value and opponent's position value
        """
        opponent = 'W' if player == 'B' else 'B'
        player_score = 0
        opponent_score = 0

        for r in range(game.rows):
            for c in range(game.cols):
                # Pieces on board edges are preferred
                if game.board[r, c] == player:
                    if r == 0 or r == game.rows - 1 or c == 0 or c == game.cols - 1:
                        player_score += 2
                    else:
                        player_score += 1
                elif game.board[r, c] == opponent:
                    if r == 0 or r == game.rows - 1 or c == 0 or c == game.cols - 1:
                        opponent_score += 2
                    else:
                        opponent_score += 1

        return player_score - opponent_score


class ClusteringHeuristic(Heuristic):
    """
    Clustering Heuristic - prefers pieces that are clustered together
    """
    def evaluate(self, game: ClobberGame, player: str) -> int:
        """
        Evaluates heuristic
        :param game: game state
        :param player: player ('B' or 'W')
        :return: The difference between player clustering value and opponent's clustering value
        """
        opponent = 'W' if player == 'B' else 'B'
        player_clustering = 0
        opponent_clustering = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(game.rows):
            for c in range(game.cols):
                if game.board[r, c] == player:
                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        if 0 <= new_r < game.rows and 0 <= new_c < game.cols and game.board[new_r, new_c] == player:
                            player_clustering += 1
                elif game.board[r, c] == opponent:
                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        if 0 <= new_r < game.rows and 0 <= new_c < game.cols and game.board[new_r, new_c] == opponent:
                            opponent_clustering += 1

        return player_clustering - opponent_clustering


class CombinedHeuristic1(Heuristic):
    """
    Combination of heuristics: 3 * MobilityHeuristic + MobilePieceCountHeuristic
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        mobility = MobilityHeuristic().evaluate(game, player)
        piece_count = MobilePieceCountHeuristic().evaluate(game, player)
        return 3 * mobility + piece_count


class CombinedHeuristic2(Heuristic):
    """
    Combination of heuristics: 2 * MobilityHeuristic + 2 * PositionHeuristic + ClusteringHeuristic
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        mobility = MobilityHeuristic().evaluate(game, player)
        position = PositionHeuristic().evaluate(game, player)
        clustering = ClusteringHeuristic().evaluate(game, player)
        return 3 * mobility + 2 * position + clustering


class CombinedHeuristic3(Heuristic):
    """
    Combination of heuristics: MobilityHeuristic + MobilePieceCountHeuristic + 3 * ClusteringHeuristic
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        mobility = MobilityHeuristic().evaluate(game, player)
        piece_count = MobilePieceCountHeuristic().evaluate(game, player)
        clustering = ClusteringHeuristic().evaluate(game, player)
        return mobility + piece_count + 3 * clustering
