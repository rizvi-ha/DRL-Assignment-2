import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import math
import time
from collections import defaultdict

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
                try:
                    feature = self.get_feature(board, pattern)
                except Exception as e:
                    print(f"Error in get_feature: {e}")
                    print(f"Pattern: {pattern}, Board: {board}")
                    exit(1)
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

# -------------------------------
# Helper function to compute the afterstate.
# It creates a copy of the environment, performs the move (without adding a new tile),
# and returns the resulting board and the immediate reward.
# -------------------------------
def compute_afterstate(env, action):
    # Create a temporary environment copy
    temp_env = Game2048Env()
    temp_env.board = np.copy(env.board)
    temp_env.score = env.score
    score_before = temp_env.score
    # Execute the move directly (using the move functions defined in Game2048Env)
    if action == 0:
        moved = temp_env.move_up()
    elif action == 1:
        moved = temp_env.move_down()
    elif action == 2:
        moved = temp_env.move_left()
    elif action == 3:
        moved = temp_env.move_right()
    else:
        moved = False
    # The reward is the increase in score due to merges
    reward = temp_env.score - score_before
    # Note: we do NOT add a random tile here (afterstate)
    return np.copy(temp_env.board), reward

def get_legal_moves(state, score):
    """
    Returns a list of legal actions (0: up, 1: down, 2: left, 3: right)
    for the given state and score by checking if the move changes the board.
    """
    moves = []
    for action in range(4):
        temp_env = Game2048Env()
        temp_env.board = state.copy()
        temp_env.score = score
        afterstate, _ = compute_afterstate(temp_env, action)
        if not np.array_equal(afterstate, state):
            moves.append(action)
    return moves

# ------------------------------
# TD-MCTS
# ------------------------------

class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        temp_env = Game2048Env()
        temp_env.board = state.copy()
        temp_env.score = score
        self.untried_actions = [a for a in range(4) if temp_env.is_move_legal(a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0

# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        if not node.children:
          return None

        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            exploitation = child.total_reward / (child.visits) if child.visits > 0 else 0.0
            exploration = self.c * math.sqrt(math.log(node.visits) / (child.visits))

            est_value = self.approximator.value(child.state)
            uct = exploitation + exploration + est_value
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        state = sim_env.board.copy()
        score = sim_env.score
        for _ in range(depth):
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            _, _, done, _ = sim_env.step(action)
            if done:
              break
        return self.approximator.value(sim_env.board)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward *= self.gamma
            node = node.parent

    def run_simulation(self, root):
        node = root
        if node is None:
            exit(1)
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            _, _, done, _ = sim_env.step(node.action)
            if done:
              break

        # Expansion
        if not sim_env.is_game_over() and node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            next_env = self.create_env_from_state(sim_env.board, sim_env.score)
            next_state, new_score, done, _ = next_env.step(action)
            child_node = TD_MCTS_Node(next_env.board.copy(), next_env.score, parent=node, action=action)
            node.children[action] = child_node
            node = child_node
            sim_env = next_env

        # Rollout
        rollout_reward = self.rollout(sim_env, self.rollout_depth)

        # Backpropagation
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

# code for submitting to eval server
# Load the N-Tuple approximator
with open('ntuple_approximator.pkl', 'rb') as f:
    value_approximator = pickle.load(f)

env = Game2048Env()

# Initialize the game environment
state = env.reset()
done = False

mcts_td = TD_MCTS(
    env,
    approximator=value_approximator,
    iterations=150,
    exploration_constant=1.41,
    rollout_depth=6,
    gamma=0.99999
)

def get_action(state, score):
    global mcts_td

    root = TD_MCTS_Node(state, score)
    for _ in range(mcts_td.iterations):
        mcts_td.run_simulation(root)

    best_act, visit_distribution = mcts_td.best_action_distribution(root)
    return best_act

def get_action_with_just_value(state, score):
    """
    Currently unused, just for testing purposes.
    """
    global value_approximator
    approximator = value_approximator

    # Load the NTupleApproximator from file if not already loaded.
    if approximator is None:
        try:
            with open("ntuple_approximator.pkl", "rb") as f:
                approximator = pickle.load(f)
        except Exception as e:
            # If loading fails, return a random action.
            return random.choice([0, 1, 2, 3])
    
    moves = get_legal_moves(state, score)
    if not moves:
        exit(1)
    
    best_action = None
    best_value = -float('inf')
    for action in moves:
        temp_env = Game2048Env()
        temp_env.board = state.copy()
        temp_env.score = score
        afterstate, immediate_reward = compute_afterstate(temp_env, action)
        # The value is the sum of the immediate reward and the approximator's estimate.
        value = immediate_reward + approximator.value(afterstate)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action

"""
if __name__ == '__main__':

    # Load the N-Tuple approximator
    with open('ntuple_approximator.pkl', 'rb') as f:
        value_approximator = pickle.load(f)

    env = Game2048Env()

    # Initialize the game environment
    state = env.reset()
    done = False

    mcts_td = TD_MCTS(
        env,
        approximator=value_approximator,
        iterations=150,
        exploration_constant=1.41,
        rollout_depth=6,
        gamma=0.99999
    )

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
