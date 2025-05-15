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


class EdgeHeuristic(Heuristic):
    """
    Edge Heuristic - prefers pieces (player's figures) on the edges of the board
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


class ImprovedPositionHeuristic(Heuristic):
    """
    Enhanced position heuristic - evaluates positions based on their tactical value
    rather than just edge preference
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        """
        Positions are valued based on:
        1. How many opponent pieces they can attack
        2. How few opponent pieces can attack them (safety)
        """
        opponent = 'W' if player == 'B' else 'B'
        player_score = 0
        opponent_score = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(game.rows):
            for c in range(game.cols):
                if game.board[r, c] == player:
                    # Attack potential (how many opponent pieces are adjacent)
                    attack_potential = 0
                    # Vulnerability (how many opponent pieces can attack this piece)
                    vulnerability = 0

                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        if 0 <= new_r < game.rows and 0 <= new_c < game.cols:
                            if game.board[new_r, new_c] == opponent:
                                attack_potential += 1
                                vulnerability += 1
                            elif game.board[new_r, new_c] == '_':  # Empty space
                                vulnerability -= 1  # Empty spaces reduce vulnerability

                    # Balance attack potential with safety
                    player_score += attack_potential * 2 - vulnerability

                elif game.board[r, c] == opponent:
                    # Do the same evaluation for opponent pieces
                    attack_potential = 0
                    vulnerability = 0

                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        if 0 <= new_r < game.rows and 0 <= new_c < game.cols:
                            if game.board[new_r, new_c] == player:
                                attack_potential += 1
                                vulnerability += 1
                            elif game.board[new_r, new_c] == '_':  # Empty space
                                vulnerability -= 1

                    opponent_score += attack_potential * 2 - vulnerability

        return player_score - opponent_score


class IsolationHeuristic(Heuristic):
    """
    Isolation Heuristic - Rewards isolating opponent pieces into separate regions
    that cannot interact.
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        """
        Evaluates heuristic based on how many isolated regions of opponent pieces exist
        More isolated regions = better for player
        """
        opponent = 'W' if player == 'B' else 'B'

        # Use a simple flood fill to identify connected regions
        def find_regions(piece_type):
            visited = set()
            regions = 0

            def flood_fill(r, c):
                if ((r, c) in visited or not (0 <= r < game.rows and 0 <= c < game.cols)
                        or game.board[r, c] != piece_type):
                    return

                visited.add((r, c))
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for dr, dc in directions:
                    flood_fill(r + dr, c + dc)

            for r in range(game.rows):
                for c in range(game.cols):
                    if game.board[r, c] == piece_type and (r, c) not in visited:
                        regions += 1
                        flood_fill(r, c)

            return regions

        player_regions = find_regions(player)
        opponent_regions = find_regions(opponent)

        # In Clobber, having your pieces in fewer regions is generally better for mobility
        # Having opponent pieces in many regions is better (more isolated/less mobile)
        return 3 * (opponent_regions - player_regions)


class EndgameHeuristic(Heuristic):
    """
    Endgame Heuristic - Focuses on creating winning patterns as the game progresses
    by evaluating the game differently based on the number of pieces left.
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        """
        Adapts evaluation based on game stage.
        Early: Mobility is key
        Middle: Positioning and trapping become important
        Endgame: Direct forcing moves and material advantage
        """
        opponent = 'W' if player == 'B' else 'B'

        # Count pieces to determine game stage
        player_pieces = 0
        opponent_pieces = 0
        total_squares = game.rows * game.cols

        for r in range(game.rows):
            for c in range(game.cols):
                if game.board[r, c] == player:
                    player_pieces += 1
                elif game.board[r, c] == opponent:
                    opponent_pieces += 1

        total_pieces = player_pieces + opponent_pieces
        filled_ratio = total_pieces / total_squares

        # Early game (more than 70% of board filled)
        if filled_ratio > 0.7:
            # Mobility is key in early game
            mobility = MobilityHeuristic().evaluate(game, player)
            return mobility * 3

        # Mid game (30-70% of board filled)
        elif filled_ratio > 0.3:
            # Balance mobility with position and trapping
            mobility = MobilityHeuristic().evaluate(game, player)
            position = EdgeHeuristic().evaluate(game, player)
            trapping = TrapHeuristic().evaluate(game, player)
            return mobility * 2 + position + trapping * 2

        # End game (less than 30% of board filled)
        else:
            # Focus on material advantage and winning positions
            mobility = MobilePieceCountHeuristic().evaluate(game, player)
            material = player_pieces - opponent_pieces

            # In the endgame, having even one more legal move than opponent is huge
            if mobility > 0:
                mobility_bonus = 10
            elif mobility < 0:
                mobility_bonus = -10
            else:
                mobility_bonus = 0

            return material * 5 + mobility_bonus


class ControlZoneHeuristic(Heuristic):
    """
    Control Zone Heuristic - Evaluates control of territories on the board
    by measuring influence zones around pieces.
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        opponent = 'W' if player == 'B' else 'B'

        # Initialize influence maps
        player_influence = [[0 for _ in range(game.cols)] for _ in range(game.rows)]
        opponent_influence = [[0 for _ in range(game.cols)] for _ in range(game.rows)]

        # Define decreasing influence based on distance
        influence_by_distance = [5, 2, 1]  # Distance 0, 1, 2
        max_distance = len(influence_by_distance) - 1

        # Calculate influence for each piece
        for r in range(game.rows):
            for c in range(game.cols):
                if game.board[r, c] == player:
                    # Spread player influence
                    for dr in range(-max_distance, max_distance + 1):
                        for dc in range(-max_distance, max_distance + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < game.rows and 0 <= nc < game.cols:
                                distance = abs(dr) + abs(dc)
                                if distance <= max_distance:
                                    player_influence[nr][nc] += influence_by_distance[distance]

                elif game.board[r, c] == opponent:
                    # Spread opponent influence
                    for dr in range(-max_distance, max_distance + 1):
                        for dc in range(-max_distance, max_distance + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < game.rows and 0 <= nc < game.cols:
                                distance = abs(dr) + abs(dc)
                                if distance <= max_distance:
                                    opponent_influence[nr][nc] += influence_by_distance[distance]

        # Calculate total control scores
        player_control = 0
        opponent_control = 0

        for r in range(game.rows):
            for c in range(game.cols):
                if player_influence[r][c] > opponent_influence[r][c]:
                    player_control += player_influence[r][c] - opponent_influence[r][c]
                elif opponent_influence[r][c] > player_influence[r][c]:
                    opponent_control += opponent_influence[r][c] - player_influence[r][c]

        return player_control - opponent_control


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


class SurvivabilityHeuristic(Heuristic):
    """
    Survivability Heuristic - Counts the player's pieces that cannot be captured by the opponent on the next move.
    """
    def evaluate(self, game: ClobberGame, player: str) -> int:
        opponent = 'W' if player == 'B' else 'B'
        opponent_moves = game.get_valid_moves(opponent)
        threatened = set((x1, y1) for (_, (x1, y1)) in opponent_moves)

        safe_pieces = 0
        for r in range(game.rows):
            for c in range(game.cols):
                if game.board[r, c] == player and (r, c) not in threatened:
                    safe_pieces += 1
        return safe_pieces


class CentralControlHeuristic(Heuristic):
    """
    Central Control Heuristic - Points for having pawns in the central squares of the board,
    which usually give more options.
    """
    def evaluate(self, game: ClobberGame, player: str) -> int:
        rows, cols = game.rows, game.cols
        center_r, center_c = rows // 2, cols // 2

        score = 0
        for r in range(rows):
            for c in range(cols):
                if game.board[r, c] == player:
                    # The closer to the center, the higher the score
                    distance = abs(r - center_r) + abs(c - center_c)
                    score += max(0, 5 - distance)  # max value 5
        return score


class TrapHeuristic(Heuristic):
    """
    Trap Heuristic - Counts positions where the opponent may be deprived of a move on the next move.
    """
    def evaluate(self, game: ClobberGame, player: str) -> int:
        opponent = 'W' if player == 'B' else 'B'
        trapped = 0
        for r in range(game.rows):
            for c in range(game.cols):
                if game.board[r, c] == opponent:
                    moves = game.get_valid_moves_from((r, c), opponent)
                    if len(moves) == 0:
                        trapped += 1
        return trapped


class OptimizedCombinedHeuristic(Heuristic):
    """
    Optimized combined heuristic that balances multiple strategic aspects
    with tuned weights
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        # Base strategic evaluation
        mobility = MobilityHeuristic().evaluate(game, player)
        mobile_pieces = MobilePieceCountHeuristic().evaluate(game, player)

        # Tactical evaluation
        position = ImprovedPositionHeuristic().evaluate(game, player)
        clustering = ClusteringHeuristic().evaluate(game, player)

        # Safety evaluation
        survivability = SurvivabilityHeuristic().evaluate(game, player)

        # Calculate total with carefully balanced weights
        return (
                3 * mobility +  # Mobility is crucial in Clobber
                2 * mobile_pieces +  # Having active pieces is important
                2 * position +  # Tactical positioning
                clustering +  # Some clustering benefit
                2 * survivability  # Survivability becomes increasingly important
        )


class StageBasedHeuristic(Heuristic):
    """
    A heuristic that changes its evaluation strategy based on the game stage
    """

    def evaluate(self, game: ClobberGame, player: str) -> int:
        opponent = 'W' if player == 'B' else 'B'

        # Count pieces to determine game stage
        player_pieces = 0
        opponent_pieces = 0
        empty_spaces = 0
        total_squares = game.rows * game.cols

        for r in range(game.rows):
            for c in range(game.cols):
                if game.board[r, c] == player:
                    player_pieces += 1
                elif game.board[r, c] == opponent:
                    opponent_pieces += 1
                else:
                    empty_spaces += 1

        total_pieces = player_pieces + opponent_pieces
        empty_ratio = empty_spaces / total_squares

        # Early game (first third)
        if empty_ratio < 0.33:
            # In early game, focus on mobility and setting up good positions
            mobility = MobilityHeuristic().evaluate(game, player) * 3
            position = ImprovedPositionHeuristic().evaluate(game, player) * 2
            return mobility + position

        # Mid game (middle third)
        elif empty_ratio < 0.66:
            # In mid game, focus on trapping and control
            mobility = MobilityHeuristic().evaluate(game, player) * 2
            trap = TrapHeuristic().evaluate(game, player) * 2
            survivability = SurvivabilityHeuristic().evaluate(game, player) * 1
            return mobility + trap + survivability

        # End game (final third)
        else:
            # In end game, prioritize winning positions and direct advantage
            mobility = MobilityHeuristic().evaluate(game, player) * 3
            mobile_pieces = MobilePieceCountHeuristic().evaluate(game, player) * 3
            isolation = IsolationHeuristic().evaluate(game, player) * 2

            if mobility > 0:  # If we have mobility advantage, emphasize it
                return mobility * 2 + mobile_pieces + isolation
            else:  # If we're behind in mobility, try to maximize piece count
                return mobility + mobile_pieces * 2 + isolation


class StyleMatchupHeuristic(Heuristic):
    """
    A meta-heuristic that combines other heuristics based on which seems to be
    performing better in the current game situation
    """

    def __init__(self):
        self.heuristics = [
            MobilityHeuristic(),
            MobilePieceCountHeuristic(),
            ImprovedPositionHeuristic(),
            ClusteringHeuristic(),
            SurvivabilityHeuristic(),
            TrapHeuristic(),
            IsolationHeuristic()
        ]

    def evaluate(self, game: ClobberGame, player: str) -> int:
        opponent = 'W' if player == 'B' else 'B'

        # Get the raw evaluations from each heuristic
        evaluations = [h.evaluate(game, player) for h in self.heuristics]

        # Find the best 3 heuristics in the current position
        best_indices = sorted(range(len(evaluations)), key=lambda i: evaluations[i], reverse=True)[:3]

        # Weight them 3, 2, 1
        weighted_sum = 3 * evaluations[best_indices[0]] + 2 * evaluations[best_indices[1]] + evaluations[
            best_indices[2]]

        return weighted_sum
