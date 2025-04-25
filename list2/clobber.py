import numpy as np
from typing import List, Tuple


class ClobberGame:
    def __init__(self, rows=5, cols=6):
        """
        Initialization of Clobber game
        :param rows: Number of rows
        :param cols: Number of columns
        """
        self.rows = rows
        self.cols = cols

        # Default starting board - black (B) on black squares, white (W) on white squares
        self.board = np.empty((rows, cols), dtype=str)
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    self.board[r, c] = 'W'  # white on white squares
                else:
                    self.board[r, c] = 'B'  # black on black squares

    def set_board(self, board):
        """
        Set the board based on the input
        :param board: board in the form of a list of list
        :return:
        """
        self.board = np.array(board)
        self.rows, self.cols = self.board.shape

    def load_board_from_input(self):
        """
        Loads a board from standard input
        :return:
        """
        print("Give the board (B - black, W - white, _ - empty:")
        board = []
        for i in range(self.rows):
            row = input().split()
            board.append(row)
        self.set_board(board)

    def print_board(self):
        """
        Prints board on standard output
        :return:
        """
        for i in range(self.rows):
            print(" ".join(self.board[i]))

    def get_valid_moves(self, player: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Generates a list of possible, valid moves for a given player
        :param player: Player ('B' or 'W')
        :return: List of tuples ((from_row, from_col), (to_row, to_col))
        """
        opponent = 'W' if player == 'B' else 'B'
        valid_moves = []

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == player:
                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        # Check if new coords are in board
                        if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                            # Check if there is an opponent on new coords
                            if self.board[new_r, new_c] == opponent:
                                valid_moves.append(((r, c), (new_r, new_c)))
        return valid_moves

    def make_move(self, move: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        Makes move on board
        :param move: tuple ((from_row, from_col), (to_row, to_col))
        :return:
        """
        (from_row, from_col), (to_row, to_col) = move
        piece = self.board[from_row, from_col]
        self.board[from_row, from_col] = '_'
        self.board[to_row, to_col] = piece

    def undo_move(self, move: Tuple[Tuple[int, int], Tuple[int, int]], opponent: str):
        """
        Undo move on board
        :param move: tuple ((from_row, from_col), (to_row, to_col))
        :param opponent: opponent, who made a move ('B' or 'W')
        :return:
        """
        (from_row, from_col), (to_row, to_col) = move
        piece = self.board[from_row, from_col]
        self.board[from_row, from_col] = piece
        self.board[to_row, to_col] = opponent

    def is_game_over(self, player: str) -> bool:
        """
        Checks if game is over for a given player
        :param player: player ('B' or 'W')
        :return: True if player doesn't have any possible moves, False otherwise
        """
        return len(self.get_valid_moves(player)) == 0
