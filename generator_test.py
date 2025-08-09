import torch
import numpy as np
from rdkit import Chem
from typing import List, Optional

from model import GenerativeModel
from data_loading import load_data, tensor_from_chars_list
from config import GenerationConfig


class SMILESGenerator:
    """SMILES molecule generator using trained LSTM model."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
        
        # Load data and vocabulary
        self.data, vocab_array = load_data(config.data_path)
        self.data = set(list(self.data))  # Convert to set for faster lookups
        self.vocab = list(vocab_array)
        
        # Initialize and load model
        self.model = GenerativeModel(
            vocab_size=len(self.vocab),
            hidden_size=config.hidden_size,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers
        )
        
        # Load trained weights
        self.model.load_state_dict(
            torch.load(config.model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
    
    def generate_batch(
        self, 
        batch_size: int, 
        temperature: float = 0.4
    ) -> List[str]:
        """Generate a batch of SMILES sequences."""
        prime_strings = [self.config.start_token] * batch_size
        
        with torch.no_grad():
            # Prepare initial input
            inputs = torch.stack([
                tensor_from_chars_list(s, self.vocab, str(self.device)).squeeze(0)
                for s in prime_strings
            ]).to(self.device)
            
            inp = inputs.unsqueeze(1)  # Shape: [batch_size, 1]
            batch_size_actual = inp.size(0)
            
            # Track completion status for each sequence
            tracker = np.array([-1] * batch_size_actual)  # -1: active, >=0: completed at position
            predicted = prime_strings.copy()
            cur_pos = 0
            completed_count = 0
            
            hidden = self.model.init_hidden(batch_size_actual, str(self.device))
            
            while completed_count < batch_size_actual and cur_pos < self.config.max_length:
                cur_pos += 1
                
                # Forward pass
                output, hidden = self.model(inp.squeeze(1), hidden)
                output_dist = output.div(temperature).exp()
                top_indices = torch.multinomial(output_dist, 1).squeeze(1)
                
                predicted_chars = [self.vocab[idx] for idx in top_indices]
                
                # Update predictions and check for completion
                for i in range(batch_size_actual):
                    if tracker[i] == -1:  # Still active
                        if predicted_chars[i] == self.config.end_token:
                            tracker[i] = cur_pos
                            completed_count += 1
                        else:
                            predicted[i] += predicted_chars[i]
                
                # Prepare next input
                if completed_count < batch_size_actual:
                    next_chars = [
                        predicted_chars[i] if tracker[i] == -1 else self.config.end_token
                        for i in range(batch_size_actual)
                    ]
                    
                    inputs = torch.stack([
                        tensor_from_chars_list(char, self.vocab, str(self.device)).squeeze(0)
                        for char in next_chars
                    ]).to(self.device)
                    inp = inputs.unsqueeze(1)
            
            # Clean up results
            final_results = []
            for idx, sequence in enumerate(predicted):
                if tracker[idx] > 0:
                    # Remove start token and truncate at end position
                    clean_seq = sequence[1:1 + tracker[idx] - 1]
                else:
                    # Remove start token only (max length reached)
                    clean_seq = sequence[1:]
                final_results.append(clean_seq)
            
            return final_results
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is chemically valid."""
        try:
            return Chem.MolFromSmiles(smiles) is not None
        except:
            return False
    
    def get_canonical_smiles(self, smiles: str) -> Optional[str]:
        """Convert SMILES to canonical form."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except:
            pass
        return None
    
    def is_novel(self, smiles: str) -> bool:
        """Check if SMILES is not in training data."""
        canonical = self.get_canonical_smiles(smiles)
        if canonical is None:
            return False
        
        formatted_smiles = f"{self.config.start_token}{canonical}{self.config.end_token}"
        return formatted_smiles not in self.data
    
    def generate_valid_novel_smiles(self, target_count: int) -> List[str]:
        """Generate valid and novel SMILES molecules."""
        valid_smiles = []
        
        with open(self.config.output_file, 'w') as output_file:
            print(f"Generating {target_count} valid novel SMILES...")
            
            while len(valid_smiles) < target_count:
                batch = self.generate_batch(self.config.n_batch, self.config.temperature)
                
                for smiles in batch:
                    if self.is_valid_smiles(smiles) and self.is_novel(smiles):
                        valid_smiles.append(smiles)
                        output_file.write(f"{smiles}\n")
                        output_file.flush()  # Ensure data is written immediately
                
                if len(valid_smiles) % 1000 == 0:
                    print(f"Generated {len(valid_smiles)} valid novel SMILES...")
        
        print(f"Generation complete! Total: {len(valid_smiles)} valid novel SMILES")
        return valid_smiles[:target_count]


def evaluate_model_quality(generator: SMILESGenerator, n_samples: int = 1000) -> dict:
    """Evaluate the quality of generated SMILES at different temperatures."""
    results = {}
    
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"Evaluating at temperature {temp}...")
        batch = generator.generate_batch(n_samples, temperature=temp)
        
        valid_count = sum(1 for s in batch if generator.is_valid_smiles(s))
        novel_count = sum(1 for s in batch if generator.is_valid_smiles(s) and generator.is_novel(s))
        
        validity = valid_count / n_samples
        novelty = novel_count / valid_count if valid_count > 0 else 0
        
        results[temp] = {
            'validity': validity,
            'novelty': novelty,
            'valid_count': valid_count,
            'novel_count': novel_count
        }
        
        print(f"Temperature {temp}: Validity={validity:.3f}, Novelty={novelty:.3f}")
    
    return results


def main():
    """Main generation script."""
    config = GenerationConfig()
    generator = SMILESGenerator(config)
    
    print(f"Loaded model from {config.model_path}")
    print(f"Vocabulary size: {len(generator.vocab)}")
    print(f"Device: {generator.device}")
    
    # Generate novel SMILES
    generator.generate_valid_novel_smiles(config.n_samples)
    
    # Optional: Evaluate model quality
    # quality_results = evaluate_model_quality(generator, n_samples=1000)
    # print("\nQuality evaluation results:")
    # for temp, metrics in quality_results.items():
    #     print(f"Temp {temp}: Validity={metrics['validity']:.3f}, Novelty={metrics['novelty']:.3f}")


if __name__ == "__main__":
    main()