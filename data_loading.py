import torch
import numpy as np
import time
import math
from random import randint
from typing import List, Tuple, Optional

def load_data(data_path: str = 'data/smiles_data.npz') -> Tuple[np.ndarray, np.ndarray]:
    """Load SMILES data and vocabulary from compressed numpy file."""
    try:
        data = np.load(data_path, allow_pickle=True)
        return data['data_set'], data['vocabs']
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in data file: {e}")

def tensor_from_chars_list(chars_list: str, vocab: List[str], device: str = 'cpu') -> torch.Tensor:
    """Convert character sequence to tensor indices."""
    tensor = torch.zeros(len(chars_list), dtype=torch.long, device=device)
    for i, char in enumerate(chars_list):
        try:
            tensor[i] = vocab.index(char)
        except ValueError:
            raise ValueError(f"Character '{char}' not found in vocabulary")
    return tensor.view(1, -1)
def process_batch(
    sequences: List[str], 
    batch_size: int, 
    vocab: List[str], 
    device: str = 'cpu'
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Create training and validation batches from sequences."""
    batches = []
    for i in range(0, len(sequences), batch_size):
        input_list = []
        output_list = []
        for j in range(i, min(i + batch_size, len(sequences))):
            input_seq = tensor_from_chars_list(sequences[j][:-1], vocab, device)
            output_seq = tensor_from_chars_list(sequences[j][1:], vocab, device)
            input_list.append(input_seq)
            output_list.append(output_seq)
        
        if input_list:  # Only create batch if we have sequences
            inp = torch.cat(input_list, 0)
            target = torch.cat(output_list, 0)
            batches.append((inp, target))
    
    train_split = int(0.9 * len(batches))
    return batches[:train_split], batches[train_split:]
def process_data_to_batches(
    data: List[str], 
    batch_size: int, 
    vocab: List[str], 
    device: str = 'cpu',
    min_length: int = 3
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Group sequences by length and create batches."""
    length_grouped_data = {}
    for sequence in data:
        seq_len = len(sequence)
        if seq_len >= min_length:
            if seq_len not in length_grouped_data:
                length_grouped_data[seq_len] = []
            length_grouped_data[seq_len].append(sequence)
    
    train_batches = []
    val_batches = []
    for sequences in length_grouped_data.values():
        train, val = process_batch(sequences, batch_size, vocab, device)
        train_batches.extend(train)
        val_batches.extend(val)
    
    return train_batches, val_batches

def get_random_batch(
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a random batch from training batches."""
    if not train_batches:
        raise ValueError("No training batches available")
    random_idx = randint(0, len(train_batches) - 1)
    return train_batches[random_idx]

def time_since(start_time: float) -> str:
    """Calculate elapsed time in minutes and seconds."""
    elapsed = time.time() - start_time
    minutes = math.floor(elapsed / 60)
    seconds = elapsed - minutes * 60
    return f'{minutes}m {seconds:.0f}s'



