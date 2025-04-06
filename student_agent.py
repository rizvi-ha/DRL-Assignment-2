import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math
import time

class MCTSNode:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = get_legal_moves(state, score)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices = [
            (child.value / (child.visits + 1e-6) + c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)), action, child)
            for action, child in self.children.items()
        ]
        return max(choices, key=lambda x: x[0])[1:]

    def expand(self):
        action = self.untried_actions.pop()
        afterstate, reward = compute_afterstate_from_state(self.state, self.score, action)
        child = MCTSNode(afterstate, self.score + reward, parent=self, action=action)
        self.children[action] = child
        return child

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def best_action(self):
        return max(self.children.items(), key=lambda x: x[1].visits)[0]

def mcts_search(state, score, approximator, time_limit=0.08):
    root = MCTSNode(state, score)
    start_time = time.time()

    while time.time() - start_time < time_limit:
        node = root

        # Selection
        while node.is_fully_expanded() and node.children:
            _, node = node.best_child()

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()

        # Simulation using NTuple value
        rollout_value = approximator.value(node.state)

        # Backpropagation
        node.backpropagate(rollout_value)

    return root.best_action()

# -------------------------------
# Transformation functions for board coordinates.
# These functions take a coordinate (r, c) on a board of size N.
# -------------------------------

def identity(coord, board_size):
    return coord

def rot90(coord, board_size):
    r, c = coord
    return (c, board_size - 1 - r)

def rot180(coord, board_size):
    r, c = coord
    return (board_size - 1 - r, board_size - 1 - c)

def rot270(coord, board_size):
    r, c = coord
    return (board_size - 1 - c, r)

def reflect_horizontal(coord, board_size):
    r, c = coord
    return (r, board_size - 1 - c)

def reflect_vertical(coord, board_size):
    r, c = coord
    return (board_size - 1 - r, c)

def reflect_main_diag(coord, board_size):
    r, c = coord
    return (c, r)

def reflect_anti_diag(coord, board_size):
    r, c = coord
    return (board_size - 1 - c, board_size - 1 - r)


# -------------------------------
# NTupleApproximator using symmetric sampling.
# -------------------------------

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Each original pattern (a list of (row, col) positions) is associated
        with a weight table (here, a defaultdict) and a set of symmetric transformations.
        """
        self.board_size = board_size
        self.patterns = patterns  # list of original n-tuple patterns
        # One weight dictionary per original pattern.
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetric groups for each pattern
        self.symmetry_groups = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_groups.append(syms)

    def generate_symmetries(self, pattern):
        """
        Generate the 8 unique symmetrical transformations of a given pattern.
        Each transformation maps a coordinate (r, c) to a new coordinate.
        The board_size is taken from self.board_size.
        """
        transforms = [identity, rot90, rot180, rot270, reflect_horizontal, reflect_vertical, reflect_main_diag, reflect_anti_diag]
        sym_set = []
        for t in transforms:
            transformed = [t(coord, self.board_size) for coord in pattern]
            # Sort the coordinates for canonical ordering.
            transformed_sorted = sorted(transformed)
            if transformed_sorted not in sym_set:
                sym_set.append(transformed_sorted)
        return sym_set

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        Empty squares (0) map to 0; otherwise, use log2.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        """
        Given a board (2D numpy array) and a list of (row, col) coordinates,
        extract the tile values, convert them using tile_to_index,
        and return the resulting tuple.
        """
        return tuple(self.tile_to_index(board[r, c]) for (r, c) in coords)

    def value(self, board):
        """
        Estimate the board value by summing (averaged over symmetry)
        the lookup table values for each original pattern.
        """
        total = 0.0
        # For each original pattern (and its weight table)
        for i, sym_group in enumerate(self.symmetry_groups):
            group_val = 0.0
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                group_val += self.weights[i][feature]
            # Average over the number of symmetric samples for this pattern.
            total += group_val / len(sym_group)
        return total

    def update(self, board, delta, alpha):
        """
        Update the weights for each pattern based on the TD error delta.
        The update is averaged over the symmetric samples.
        """
        for i, sym_group in enumerate(self.symmetry_groups):
            # Distribute the update equally over the symmetry group.
            update_amount = alpha * delta / len(sym_group)
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                self.weights[i][feature] += update_amount



# ------------------------------
# Game2048 Environment (provided starting code)
# ------------------------------
class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]
        self.last_move_valid = True  # Record if the last move was valid
        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False
        self.last_move_valid = moved
        if moved:
            self.add_random_tile()
        done = self.is_game_over()
        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        COLOR_MAP = {0:"#3c3a32", 2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563",
                     32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", 512:"#edc850",
                     1024:"#edc53f", 2048:"#edc22e"}
        TEXT_COLOR = {0:"white", 2:"black", 4:"black", 8:"white", 16:"white",
                      32:"white", 64:"white", 128:"white", 256:"white", 512:"white",
                      1024:"white", 2048:"white"}
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row
        
    def is_move_legal(self, action):
            temp_board = self.board.copy()

            if action == 0:  # Move up
                for j in range(self.size):
                    col = temp_board[:, j]
                    new_col = self.simulate_row_move(col)
                    temp_board[:, j] = new_col
            elif action == 1:  # Move down
                for j in range(self.size):
                    col = temp_board[:, j][::-1]
                    new_col = self.simulate_row_move(col)
                    temp_board[:, j] = new_col[::-1]
            elif action == 2:  # Move left
                for i in range(self.size):
                    row = temp_board[i]
                    temp_board[i] = self.simulate_row_move(row)
            elif action == 3:  # Move right
                for i in range(self.size):
                    row = temp_board[i][::-1]
                    new_row = self.simulate_row_move(row)
                    temp_board[i] = new_row[::-1]
            else:
                raise ValueError("Invalid action")
            return not np.array_equal(self.board, temp_board)

def compute_afterstate_from_state(state, score, action):
    """
    Given the current board (state) and score, simulate performing the move (action)
    deterministically (i.e. without adding a random tile) and return the afterstate board
    and the immediate reward (increase in score).
    """
    temp_env = Game2048Env()
    temp_env.board = np.copy(state)
    temp_env.score = score
    score_before = temp_env.score
    if action == 0:
        temp_env.move_up()
    elif action == 1:
        temp_env.move_down()
    elif action == 2:
        temp_env.move_left()
    elif action == 3:
        temp_env.move_right()
    else:
        # Should not happen; return the state unchanged.
        return np.copy(state), 0
    reward = temp_env.score - score_before
    return np.copy(temp_env.board), reward

def get_legal_moves(state, score):
    """
    Returns a list of legal actions (0: up, 1: down, 2: left, 3: right)
    for the given state and score by checking if the move changes the board.
    """
    moves = []
    for action in range(4):
        afterstate, _ = compute_afterstate_from_state(state, score, action)
        if not np.array_equal(afterstate, state):
            moves.append(action)
    return moves

# code for submitting to eval server
# Load the N-Tuple approximator
with open('ntuple_approximator.pkl', 'rb') as f:
    approximator = pickle.load(f)

env = Game2048Env()

# Initialize the game environment
state = env.reset()

def get_action(state, score):
    global approximator
    if approximator is None:
        try:
            with open("ntuple_approximator.pkl", "rb") as f:
                approximator = pickle.load(f)
        except Exception:
            return random.choice([0, 1, 2, 3])

    moves = get_legal_moves(state, score)
    if not moves:
        return random.choice([0, 1, 2, 3])

    # MCTS with NTuple evaluation
    return mcts_search(state, score, approximator, time_limit=0.3)

"""
if __name__ == '__main__':
    # Load the N-Tuple approximator
    with open('ntuple_approximator.pkl', 'rb') as f:
        approximator = pickle.load(f)

    env = Game2048Env()

    # Initialize the game environment
    state = env.reset()
    done = False

    i = 0 
    start = time.time()
    while not done:
        if i % 10 == 0:
            print(f"Score: {env.score}")
            print(f"Time: {time.time() - start:.2f} seconds")

        action = get_action(state, env.score)

        # Execute the action in the environment
        state, reward, done, _ = env.step(action)
        i += 1

    print(f"Game Over! Final Score: {env.score}")
"""
