import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import os
from student_agent import Game2048Env, NTupleApproximator
from tqdm import tqdm

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

# -------------------------------
# TD Learning for 2048 using afterstate evaluation.
# -------------------------------

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.
    Uses afterstate evaluation: for each legal move, the value is computed as
    (immediate reward from the move) + (approximator value of the resulting afterstate).
    The TD error is computed using the best subsequent afterstate value.
    """
    final_scores = []
    success_flags = []
    # Open log file to save the results
    log_file = open("td_learning_log.txt", "w")

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        previous_score = 0
        done = False
        max_tile = np.max(state)
        current_afterstate = None  # will store afterstate from the chosen action

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # Action selection (ε-greedy)
            if random.random() < epsilon:
                action = random.choice(legal_moves)
                current_afterstate, immediate_reward = compute_afterstate(env, action)
            else:
                best_value = -float('inf')
                best_action = None
                best_afterstate = None
                best_reward = 0
                for a in legal_moves:
                    afterstate, reward = compute_afterstate(env, a)
                    value_est = reward + approximator.value(afterstate)
                    if value_est > best_value:
                        best_value = value_est
                        best_action = a
                        best_afterstate = afterstate
                        best_reward = reward
                action = best_action
                current_afterstate = best_afterstate
                immediate_reward = best_reward

            # Take the action in the real environment (which adds a random tile)
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TD Learning update:
            # For afterstate TD-learning, if not terminal, we compute the best next afterstate value.
            if not done:
                next_legal_moves = [a for a in range(4) if env.is_move_legal(a)]
                if next_legal_moves:
                    best_next_value = -float('inf')
                    for a_next in next_legal_moves:
                        next_afterstate, r_next = compute_afterstate(env, a_next)
                        val = r_next + approximator.value(next_afterstate)
                        if val > best_next_value:
                            best_next_value = val
                else:
                    best_next_value = 0.0
            else:
                best_next_value = 0.0

            # Compute TD error: (target - current value)
            # Here target = best_next_value (which already includes the immediate reward from the next move)
            current_value = approximator.value(current_afterstate)
            delta = best_next_value - current_value
            approximator.update(current_afterstate, delta, alpha)

            # Move on to the next state.
            state = next_state

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
            log_file.write(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}\n")
            log_file.flush()

    # CLose the log file 
    log_file.close()

    return final_scores

# -------------------------------
# Define a few n-tuple patterns.
# -------------------------------

# Horizontal 4-tuples: each row
patterns = [[(i, j) for j in range(4)] for i in range(4)]
# Vertical 4-tuples: each column
patterns += [[(j, i) for j in range(4)] for i in range(4)]
# A 2x2 square pattern (top-left)
patterns.append([(0,0), (0,1), (1,0), (1,1)])

# Additional diverse patterns
patterns += [
    [(0, 0), (1, 1), (2, 2), (3, 3)],
    [(0, 3), (1, 2), (2, 1), (3, 0)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(2, 2), (2, 3), (3, 3), (3, 2)],
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(2, 2), (3, 2), (3, 3), (2, 3)]
]

# Initialize the approximator and the game environment. If already exists pickled version, load
if os.path.exists("ntuple_approximator.pkl"):
    with open("ntuple_approximator.pkl", "rb") as f:
        approximator = pickle.load(f)
    print("Loaded existing NTupleApproximator from file.")
else:
    approximator = NTupleApproximator(board_size=4, patterns=patterns)

# Assume Game2048Env is defined as provided.
env = Game2048Env()

# Run TD-Learning training.
# For quick testing we use 1,000 episodes; for stronger performance more episodes are needed.
final_scores = td_learning(env, approximator, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1)

# Save the trained approximator to a file.
with open("ntuple_approximator.pkl", "wb") as f:
    pickle.dump(approximator, f)

