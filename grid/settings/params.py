from typing import NamedTuple


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    render_each: int = 1000  # Render the environment every n episodes - set to 0 to disable rendering
    save_qtable: bool = True  # Whether to save the Q-table after training
    qtable_path: str = "qtable.npy"  # Path to save/load the Q-table
    log_path: str = "training_log.csv"  # Path to save the training log
    plot_path: str = "training_plot.png"  # Path to save the training plot
    verbose: bool = True  # Whether to print training progress