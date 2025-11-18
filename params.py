from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training parameters.
    
    This class encapsulates all training-related configuration including
    episode counts, hyperparameters, rendering, and persistence settings.
    """
    total_episodes: int = 1000
    n_runs: int = 1
    learning_rate: float = 0.1
    gamma: float = 0.99
    seed: Optional[int] = None
    render_each: int = 0  # Render every nth run (0 to disable)
    render_delay: float = 0.05  # Seconds between frames when rendering
    save_model: bool = True
    model_path: Optional[Path] = None
    log_path: Optional[Path] = None
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.total_episodes <= 0:
            raise ValueError("total_episodes must be positive")
        if self.n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.render_each < 0:
            raise ValueError("render_each must be non-negative")
        
        # Convert string paths to Path objects
        if self.model_path is not None and not isinstance(self.model_path, Path):
            self.model_path = Path(self.model_path)
        if self.log_path is not None and not isinstance(self.log_path, Path):
            self.log_path = Path(self.log_path)


@dataclass
class EnvironmentConfig:
    """Base configuration for environment settings.
    
    This class provides a flexible container for environment-specific
    parameters through the params dictionary.
    """
    state_size: Optional[int] = None
    action_size: Optional[int] = None
    params: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.state_size is not None and self.state_size <= 0:
            raise ValueError("state_size must be positive")
        if self.action_size is not None and self.action_size <= 0:
            raise ValueError("action_size must be positive")


@dataclass
class FrozenLakeConfig(EnvironmentConfig):
    """Configuration specific to FrozenLake environment.
    
    Extends EnvironmentConfig with FrozenLake-specific parameters.
    """
    map_size: int = 4
    is_slippery: bool = True
    proba_frozen: float = 0.9
    
    def __post_init__(self):
        """Validate FrozenLake-specific configuration."""
        super().__post_init__()
        
        if self.map_size <= 0:
            raise ValueError("map_size must be positive")
        if not 0 <= self.proba_frozen <= 1:
            raise ValueError("proba_frozen must be between 0 and 1")
        
        # Set state_size and action_size based on map_size
        if self.state_size is None:
            self.state_size = self.map_size * self.map_size
        if self.action_size is None:
            self.action_size = 4  # FrozenLake always has 4 actions (up, down, left, right)
        
        # Store FrozenLake params in the params dict for compatibility
        self.params.update({
            'map_size': self.map_size,
            'is_slippery': self.is_slippery,
            'proba_frozen': self.proba_frozen
        })


# Legacy Params class for backwards compatibility
@dataclass
class Params:
    """Legacy parameter class for backwards compatibility.
    
    This class is deprecated. Use TrainingConfig and FrozenLakeConfig instead.
    It combines training and environment parameters in a single class.
    """
    total_episodes: int = 1000
    learning_rate: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    map_size: int = 4
    seed: int = 42
    is_slippery: bool = True
    n_runs: int = 1
    action_size: int = 4
    state_size: int = 16  # 4x4 grid
    proba_frozen: float = 0.9
    render_each: int = 0
    render_delay: float = 0.05
    save_qtable: bool = True
    qtable_path: str = "qtable.npy"
    log_path: str = "training_log.csv"
    plot_path: str = "training_plot.png"
    verbose: bool = True
    
    @classmethod
    def from_configs(cls, training: TrainingConfig, env: FrozenLakeConfig, epsilon: float = 0.1):
        """Create a Params instance from new-style configs.
        
        Args:
            training: TrainingConfig instance
            env: FrozenLakeConfig instance
            epsilon: Exploration parameter
            
        Returns:
            Params instance for backwards compatibility
        """
        return cls(
            total_episodes=training.total_episodes,
            learning_rate=training.learning_rate,
            gamma=training.gamma,
            epsilon=epsilon,
            map_size=env.map_size,
            seed=training.seed or 42,
            is_slippery=env.is_slippery,
            n_runs=training.n_runs,
            action_size=env.action_size or 4,
            state_size=env.state_size or 16,
            proba_frozen=env.proba_frozen,
            render_each=training.render_each,
            render_delay=training.render_delay,
            save_qtable=training.save_model,
            qtable_path=str(training.model_path) if training.model_path else "qtable.npy",
            log_path=str(training.log_path) if training.log_path else "training_log.csv",
            verbose=training.verbose
        )