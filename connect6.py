import sys
import numpy as np
import random
import math
import time

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
          0 - No winner yet
          1 - Black wins
          2 - White wins
        """
        # Use the same check as in check_win_board below.
        return check_win_board(self.board)

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a move using a Monte Carlo Tree Search (MCTS) approach."""
        if self.game_over:
            print("? Game over")
            return

        # If board is completely empty, play the center (first move: one stone)
        if np.count_nonzero(self.board) == 0:
            center = self.size // 2
            move_str = f"{self.index_to_label(center)}{center+1}"
            self.play_move(color, move_str)
            print(f"{move_str}\n\n", end='', flush=True)
            print(move_str, file=sys.stderr)
            return move_str

        # Run MCTS search from the current state; you can adjust itermax as needed.
        best_move = mcts_search(self.board, self.turn, itermax=1000)
        # Convert the move (a list of (row, col) tuples) to the GTP move string.
        move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in best_move)
        self.play_move(color, move_str)
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)
        return move_str

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

# ----- MCTS Helper Functions and Classes -----

def check_win_board(board):
    """Checks for a win on the given board.
    Returns 0 if no winner, otherwise returns the winning color (1 or 2)."""
    size = board.shape[0]
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(size):
        for c in range(size):
            if board[r, c] != 0:
                current_color = board[r, c]
                for dr, dc in directions:
                    # Ensure not counting backwards.
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < size and 0 <= prev_c < size and board[prev_r, prev_c] == current_color:
                        continue
                    count = 0
                    rr, cc = r, c
                    while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == current_color:
                        count += 1
                        rr += dr
                        cc += dc
                    if count >= 6:
                        return current_color
    return 0

def get_legal_moves(board, turn):
    """Generates legal moves for the current board state.
    For an empty board, returns a one-stone move (center).
    Otherwise, returns candidate moves as lists of two positions (each position is a (row, col) tuple).
    Moves are generated from empty positions near existing stones."""
    size = board.shape[0]
    num_stones = np.count_nonzero(board)
    # For an empty board, one move: the center.
    if num_stones == 0:
        return [[(size // 2, size // 2)]]

    # Gather candidate positions from neighbors of existing stones.
    indices = np.argwhere(board != 0)
    candidates = set()
    for (r, c) in indices:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                    candidates.add((nr, nc))
    candidates = list(candidates)
    # Limit the candidates to a maximum number to reduce branching.
    if len(candidates) > 20:
        candidates = random.sample(candidates, 20)

    moves = []
    # After the first move, each turn plays two stones.
    # Generate all unique pairs from the candidate positions.
    if len(candidates) >= 2:
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                moves.append([candidates[i], candidates[j]])
    # If for some reason there is only one candidate, fallback to a one-stone move.
    if not moves:
        moves = [[candidates[0]]]
    return moves

def apply_move(board, move, turn):
    """Applies a move (a list of positions) to the board copy and returns the modified board."""
    for (r, c) in move:
        board[r, c] = turn
    return board

def best_child(node, c_param=1.4):
    """Returns the child of node with the highest UCT value."""
    choices = []
    for child in node.children:
        if child.visits == 0:
            # Favor unvisited children.
            uct = float('inf')
        else:
            uct = child.wins / child.visits + c_param * math.sqrt(math.log(node.visits) / child.visits)
        choices.append(uct)
    return node.children[np.argmax(choices)]

def rollout(board, turn):
    """Simulates a random playout from the given board state.
    Returns the winning color (1 or 2) or 0 for a draw."""
    current_turn = turn
    board = board.copy()
    while True:
        winner = check_win_board(board)
        if winner != 0:
            return winner
        legal_moves = get_legal_moves(board, current_turn)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board = apply_move(board, move, current_turn)
        current_turn = 3 - current_turn
    return 0  # Treat draw as 0.

class MCTSNode:
    def __init__(self, board, turn, move=None, parent=None):
        self.board = board  # Numpy array representing the board state.
        self.turn = turn    # Which player's turn at this node.
        self.move = move    # The move (list of positions) that led to this node.
        self.parent = parent
        self.children = []
        self.untried_moves = get_legal_moves(board, turn)
        self.visits = 0
        self.wins = 0

def mcts_search(root_board, root_turn, itermax=1000):
    """Performs an MCTS search starting from the given board state and turn.
    Returns the move (a list of (row, col) positions) with the highest visit count."""
    root_node = MCTSNode(root_board.copy(), root_turn)
    for _ in range(itermax):
        node = root_node
        board_copy = root_board.copy()
        turn = root_turn

        # Selection: traverse the tree using best_child until a node with untried moves is found.
        while node.untried_moves == [] and node.children != []:
            node = best_child(node)
            board_copy = apply_move(board_copy, node.move, turn)
            turn = 3 - turn

        # Expansion: if node has untried moves, expand one.
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            board_copy = apply_move(board_copy, move, turn)
            child_node = MCTSNode(board_copy.copy(), 3 - turn, move=move, parent=node)
            node.children.append(child_node)
            node = child_node
            turn = 3 - turn

        # Simulation: rollout from the expanded node.
        result = rollout(board_copy.copy(), turn)

        # Backpropagation: update node statistics.
        while node is not None:
            node.visits += 1
            # Here we backpropagate a win (reward 1) if the simulation result equals the root's turn.
            if result == root_turn:
                node.wins += 1
            node = node.parent

    # Select the child with the highest visits.
    best = max(root_node.children, key=lambda n: n.visits)
    return best.move

# ----- End of MCTS helper functions -----

if __name__ == "__main__":
    game = Connect6Game()
    game.run()

