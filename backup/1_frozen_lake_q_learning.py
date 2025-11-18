import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt


def plot_q_table(q_table, observation_space_n, filename="q_table_actions.png"):
    """
    Visualize Q-table as a heatmap with arrows showing best action.
    
    Args:
        q_table: Q-table array with shape (n_states, n_actions)
        observation_space_n: Number of states in the environment
        filename: Output filename for the plot
    """
    # Calculate grid dimensions
    grid_size = int(np.sqrt(observation_space_n))
    
    # Create a grid to store the max Q-value for each state (for heatmap)
    max_q_grid = np.zeros((grid_size, grid_size))
    
    # Create a grid to store the best action for each state
    best_action_grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Fill grids
    for state in range(observation_space_n):
        row, col = divmod(state, grid_size)
        max_q_grid[row, col] = np.max(q_table[state])
        best_action_grid[row, col] = np.argmax(q_table[state])
    
    # Find optimal path from start (state 0) following best actions
    optimal_path = []
    current_state = 0
    visited = set()
    max_steps = observation_space_n  # Prevent infinite loops
    
    # Action to state transitions
    action_deltas = {
        0: (0, -1),  # Left
        1: (1, 0),   # Down
        2: (0, 1),   # Right
        3: (-1, 0)   # Up
    }
    
    for _ in range(max_steps):
        if current_state in visited:
            break  # Avoid loops
        
        visited.add(current_state)
        optimal_path.append(current_state)
        
        # Check if we reached a terminal state (no positive Q-values means terminal)
        if np.max(q_table[current_state]) <= 0:
            break
        
        # Get best action
        best_action = np.argmax(q_table[current_state])
        
        # Calculate next state
        row, col = divmod(current_state, grid_size)
        dr, dc = action_deltas[best_action]
        new_row, new_col = row + dr, col + dc
        
        # Check bounds
        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
            current_state = new_row * grid_size + new_col
        else:
            break  # Out of bounds
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot heatmap
    im = ax.imshow(max_q_grid, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Q-value', rotation=270, labelpad=20)
    
    # Draw optimal path with green line
    if len(optimal_path) > 1:
        path_x = []
        path_y = []
        for state in optimal_path:
            row, col = divmod(state, grid_size)
            path_x.append(col)
            path_y.append(row)
        
        ax.plot(path_x, path_y, color='lime', linewidth=4,
               marker='o', markersize=8, alpha=0.7, label='Optimal Path')
    
    # Action arrows
    action_arrows = {
        0: '←',  # Left
        1: '↓',  # Down
        2: '→',  # Right
        3: '↑'   # Up
    }
    
    # Draw arrows for best action in each cell
    for state in range(observation_space_n):
        row, col = divmod(state, grid_size)
        
        # Only show arrow if state has been explored
        if np.max(q_table[state]) > 0:
            best_action = best_action_grid[row, col]
            arrow = action_arrows[best_action]
            
            # Draw arrow
            ax.text(col, row, arrow, ha='center', va='center',
                   fontsize=20, color='black', weight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add legend and title
    if len(optimal_path) > 1:
        ax.legend(loc='upper right')
    
    states_explored = np.sum(np.max(q_table, axis=1) > 0)
    ax.set_title(
        f'Q-table Policy Heatmap ({grid_size}×{grid_size})\n'
        f'Arrow = Best Action, Color = Max Q-value, Green = Optimal Path\n'
        f'States explored: {states_explored}/{observation_space_n}',
        fontsize=14, pad=20
    )
    
    # Add grid
    ax.set_xticks(np.arange(grid_size) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nQ-table heatmap saved to '{filename}'")
    print(f"Grid size: {grid_size}×{grid_size} ({observation_space_n} states)")
    print(f"States explored: {states_explored}/{observation_space_n} ({states_explored/observation_space_n*100:.1f}%)")


def run(
        is_training = True, # If is training, we update the Q-table and avoid logging to better performance; if not, we just use it to play
        episodes = 1000,
        render = True,
        map_name = "8x8",
        is_slippery = True,
):
    # slippery means that the agent may not always move in the intended direction - it might slip to the side
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, map_name=map_name, render_mode="human" if render else None)

    print("Environment info:")
    print(f"- Action space: {env.action_space}")
    print(f"- Observation space: {env.observation_space}")
    print(f"- Spec: {env.spec.to_json()}")

    # Q-learning

    if is_training:
        ## Creating a Q-table to store Q-values for each state-action pair - all initialized to zero
        q_table = np.zeros((env.observation_space.n, env.action_space.n)) # 64 states and 4 actions
        print(f"\nInitialize Q-table for training with shape: {q_table.shape}")
    else:
        ## Load pre-trained Q-table from file
        q_table = np.load("frozen_lake_q_table.npy")
        print(f"\nLoaded Q-table for playing with shape: {q_table.shape}")

    ## Hyperparameters
    learning_rate_alpha = 0.9       # Alpha or learning rate - how much new information overrides old
    discount_factor_gamma = 0.9     # Gamma or discount factor - how much future rewards are considered 

    # Epsilon-greedy policy
    ## Epsilon determines the exploration-exploitation trade-off
    epsilon = 1                 # 100% of the time we explore (random action), 0% we exploit (best known action) - start with pure exploration to build Q-table
    epsilon_decay_rate = 0.0001  # Decay rate for epsilon after each episode - gradually shift from exploration to exploitation
    # This means that we need at least 10000 episodes to shift from pure exploration to pure exploitation (epsilon=0)
    
    rng = np.random.default_rng() # Random number generator for action selection

    # Rewards per episode
    rewards_per_episode = np.zeros(episodes)  # To track rewards per episode

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")
        state = env.reset()[0] # States: 0 to 63 for 8x8 grid (total 64 states, 0 is start on top left and 63 is goal on bottom right)
        if not is_training:
            print(f"\nTable state: {state}")

        terminated = False # True when fall in hole or reach goal
        truncated = False # True when reach max steps (100 by default)

        while not (terminated or truncated):

            if is_training and rng.random() < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()  # Random action
                print(f"\nTaking random action (exploration): {action}") # Actions: 0=Left, 1=Down, 2=Right, 3=Up
            else:
                # Exploit: choose the action with highest Q-value for current state
                action = np.argmax(q_table[state])
                if not is_training:
                    print(f"\nTaking action based on Q-table (exploitation): {action}") # Actions: 0=Left, 1=Down, 2=Right, 3=Up
                    print(f"- Q-values for state {state}: {q_table[state]}") # Q-values for current state
                    print(f"- Action probabilities for each action: {q_table[state] / np.sum(q_table[state]) if np.sum(q_table[state]) > 0 else 'N/A (all zeros)'}")

            new_state, reward, terminated, truncated, info = env.step(action)
            if not is_training:
                print(f"New State: {new_state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

            if is_training:
                # Q-learning update rule
                # Formula -> Q(s, a) = Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
                q_table[state, action] = q_table[state, action] + learning_rate_alpha * (
                    reward + discount_factor_gamma * np.max(q_table[new_state]) - q_table[state, action]
                )

            env.render()

            state = new_state # Update state to new state for next iteration

        epsilon = max(0, epsilon - epsilon_decay_rate) # Decay epsilon but ensure it doesn't go below 0
        print(f"- Epsilon after episode {episode + 1}: {epsilon}")

        if (epsilon == 0):
            print("Epsilon has reached 0, switching to pure exploitation.")
            print("Lowering the learning rate to 0.0001 to stabilize learning.")
            learning_rate_alpha = 0.0001

        if (reward > 0):
            rewards_per_episode[episode] = reward
            print(f"Episode {episode + 1} finished with reward: {reward}")

    env.close()

    # Visualize Q-table policy
    plot_q_table(q_table, env.observation_space.n, "q_table_actions.png")

    # Create second figure for rewards plot
    plt.figure(2)
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.title("Sum of Rewards over Episodes (100 episode window)")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of Rewards")
    plt.savefig("q_table_rewards_over_episodes.png")

    if is_training:
        # Saving the Q-table for future use
        np.save("frozen_lake_q_table.npy", q_table)
        print("\nQ-table saved to 'frozen_lake_q_table.npy'")



if __name__ == "__main__":
    run(episodes=30000, render=False, map_name="8x8", is_slippery=False, is_training=True) # Training without slippery
    # run(episodes=15000, render=False, map_name="8x8", is_slippery=True, is_training=True) # Training with slippery
    # run(episodes=1000, render=False, map_name="8x8", is_slippery=False, is_training=False) # See the graphs generated from training
    # run(episodes=2, render=True, map_name="8x8", is_slippery=False, is_training=False) # Playing with rendering