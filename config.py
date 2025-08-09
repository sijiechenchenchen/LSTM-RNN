from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration class for model training parameters."""
    
    # Model parameters
    hidden_size: int = 1024
    embedding_dim: int = 248
    n_layers: int = 3
    dropout: float = 0.2
    
    # Training parameters
    batch_size: int = 128
    learning_rate: float = 0.005
    n_batches: int = 200000
    
    # Logging and saving
    print_every: int = 100
    plot_every: int = 10
    save_every: int = 1000
    
    # Device settings
    device: str = 'cpu'
    
    # Data parameters
    data_path: str = 'data/smiles_data.npz'
    results_folder: str = 'results/'
    
    # Special tokens
    start_token: str = '!'
    end_token: str = ' '
    
    # Generation parameters
    max_length: int = 200
    temperature: float = 0.4


@dataclass
class GenerationConfig:
    """Configuration class for model generation parameters."""
    
    # Model parameters (should match training config)
    hidden_size: int = 1024
    embedding_dim: int = 248
    n_layers: int = 3
    
    # Generation parameters
    n_samples: int = 1000000
    n_batch: int = 1000
    temperature: float = 0.4
    max_length: int = 200
    
    # Device settings
    device: str = 'cpu'
    
    # File paths
    model_path: str = 'mytraining.pt'
    data_path: str = 'data/smiles_data.npz'
    output_file: str = 'results/generated_smiles_non-redundant.txt'
    
    # Special tokens
    start_token: str = '!'
    end_token: str = ' '