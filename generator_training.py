import torch
import torch.nn as nn
import time
import os
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional

from model import GenerativeModel
from data_loading import load_data, process_data_to_batches, get_random_batch, time_since, tensor_from_chars_list
from config import TrainingConfig


def setup_training() -> Tuple[TrainingConfig, GenerativeModel, torch.optim.Optimizer, nn.CrossEntropyLoss]:
    """Initialize training configuration and components."""
    config = TrainingConfig()
    
    # Load data
    data, vocab_array = load_data(config.data_path)
    vocab = list(vocab_array)
    vocab_size = len(vocab)
    
    # Initialize model
    model = GenerativeModel(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        embedding_dim=config.embedding_dim,
        n_layers=config.n_layers,
        dropout=config.dropout
    )
    
    # Move to device if available
    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    model = model.to(device)
    config.device = str(device)
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    return config, model, optimizer, criterion, vocab, data


def prepare_data(data, config, vocab):
    """Prepare training and validation batches."""
    print("Processing batches...")
    train_batches, val_batches = process_data_to_batches(
        data, config.batch_size, vocab, config.device
    )
    print(f"Finished processing batches. Train: {len(train_batches)}, Val: {len(val_batches)}")
    return train_batches, val_batches


def train_step(
    model: GenerativeModel, 
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    inp: torch.Tensor, 
    target: torch.Tensor,
    device: str
) -> float:
    """Perform a single training step."""
    model.train()
    batch_size = inp.size(0)
    sequence_length = inp.size(1)
    
    hidden = model.init_hidden(batch_size, device)
    model.zero_grad()
    
    total_loss = 0
    for t in range(sequence_length):
        output, hidden = model(inp[:, t], hidden)
        loss = criterion(output, target[:, t])
        total_loss += loss
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item() / sequence_length


def evaluate(
    model: GenerativeModel,
    vocab: list,
    config: TrainingConfig,
    prime_str: str = '!'
) -> str:
    """Generate a sample sequence from the model."""
    model.eval()
    
    with torch.no_grad():
        inp = tensor_from_chars_list(prime_str, vocab, config.device)
        batch_size = inp.size(0)
        hidden = model.init_hidden(batch_size, config.device)
        predicted = prime_str
        
        for _ in range(config.max_length):
            output, hidden = model(inp, hidden)
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(config.temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            # Add predicted character to string and use as next input
            predicted_char = vocab[top_i]
            
            if predicted_char == config.end_token:
                break
                
            predicted += predicted_char
            inp = tensor_from_chars_list(predicted_char, vocab, config.device)
            
        return predicted


def train_model():
    """Main training loop."""
    config, model, optimizer, criterion, vocab, data = setup_training()
    train_batches, val_batches = prepare_data(data, config, vocab)
    
    # Create results directory
    os.makedirs(f"{config.results_folder}models", exist_ok=True)
    os.makedirs(f"{config.results_folder}logs", exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(f"{config.results_folder}logs")
    
    start_time = time.time()
    all_losses = []
    loss_avg = 0
    
    print(f"Starting training on {config.device}...")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training batches: {len(train_batches)}")
    
    for batch_idx in range(1, config.n_batches + 1):
        inp, target = get_random_batch(train_batches)
        loss = train_step(model, optimizer, criterion, inp, target, config.device)
        
        writer.add_scalar('Training/Loss', loss, batch_idx)
        loss_avg += loss
        
        if batch_idx % config.print_every == 0:
            progress = batch_idx / config.n_batches * 100
            elapsed = time_since(start_time)
            print(f'[{elapsed} ({batch_idx} {progress:.0f}%) {loss:.4f}]')
            sample = evaluate(model, vocab, config, config.start_token)
            print(f'{sample}\n')
        
        if batch_idx % config.plot_every == 0:
            all_losses.append(loss_avg / config.plot_every)
            loss_avg = 0
        
        if batch_idx % config.save_every == 0 or batch_idx == 1:
            model_path = f"{config.results_folder}models/mytraining_{batch_idx}.pt"
            torch.save(model.state_dict(), model_path)
            print(f'[Debug] Model saved to {model_path}')
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    train_model()