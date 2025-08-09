# LSTM-RNN for SMILES Generation

A PyTorch implementation of an LSTM-based generative model for creating novel molecular SMILES (Simplified Molecular-Input Line-Entry System) strings. This project uses recurrent neural networks to learn patterns in chemical structures and generate new, chemically valid molecules.

## Features

- **LSTM-based generative model** for molecular SMILES generation
- **Configurable training parameters** through configuration classes
- **Batch processing** with length-based grouping for efficient training
- **Novel molecule generation** with validity and novelty checking
- **Post-processing tools** for filtering and analysis
- **TensorBoard integration** for training visualization
- **Modern PyTorch implementation** with type hints and error handling

## Project Structure

```
LSTM-RNN/
├── model.py              # Shared LSTM generative model
├── config.py             # Configuration classes for training and generation
├── data_loading.py       # Data loading and preprocessing utilities
├── data_processing.py    # ChEMBL data processing script
├── generator_training.py # Model training script
├── generator_test.py     # Model inference and SMILES generation
├── post_process.py       # Post-processing and filtering utilities
└── data/                 # Data directory (create manually)
    ├── chembl_smiles.txt  # Raw ChEMBL SMILES data
    └── smiles_data.npz    # Processed training data
```

## Requirements

Install the required dependencies:

```bash
pip install torch numpy rdkit tensorboard
```

### Dependencies
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **RDKit** - Cheminformatics toolkit for SMILES validation
- **TensorBoard** - Training visualization

## Quick Start

### 1. Data Preparation

First, prepare your SMILES data:

```bash
# Place your ChEMBL SMILES data in data/chembl_smiles.txt
# Then process the data:
python data_processing.py
```

This will create `data/smiles_data.npz` with processed SMILES and vocabulary.

### 2. Training

Train the LSTM model:

```bash
python generator_training.py
```

Training parameters can be modified in `config.py`:
- `batch_size`: Training batch size (default: 128)
- `learning_rate`: Learning rate (default: 0.005)
- `n_batches`: Number of training batches (default: 200,000)
- `hidden_size`: LSTM hidden size (default: 1024)

### 3. Generation

Generate novel SMILES molecules:

```bash
python generator_test.py
```

This will generate valid, novel SMILES and save them to `results/generated_smiles_non-redundant.txt`.

### 4. Post-processing

Filter generated molecules:

```bash
python post_process.py
```

## Configuration

### Training Configuration (`TrainingConfig`)

```python
@dataclass
class TrainingConfig:
    # Model parameters
    hidden_size: int = 1024
    embedding_dim: int = 248
    n_layers: int = 3
    
    # Training parameters
    batch_size: int = 128
    learning_rate: float = 0.005
    n_batches: int = 200000
    
    # File paths
    data_path: str = 'data/smiles_data.npz'
    results_folder: str = 'results/'
```

### Generation Configuration (`GenerationConfig`)

```python
@dataclass
class GenerationConfig:
    # Generation parameters
    n_samples: int = 1000000
    n_batch: int = 1000
    temperature: float = 0.4
    
    # File paths
    model_path: str = 'mytraining.pt'
    output_file: str = 'results/generated_smiles_non-redundant.txt'
```

## Model Architecture

The generative model consists of:

1. **Embedding Layer** - Converts character indices to dense vectors
2. **LSTM Layers** - 3-layer LSTM with dropout for sequence modeling
3. **Linear Output Layer** - Maps hidden states to vocabulary probabilities

```python
class GenerativeModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, n_layers):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=0.2)
        self.linear = nn.Linear(hidden_size, vocab_size)
```

## Data Format

SMILES strings are processed with special tokens:
- `!` - Start token
- ` ` (space) - End token

Example: `!CCO ` represents the ethanol molecule `CCO`.

## Evaluation Metrics

The model generates molecules evaluated on:
- **Validity** - Percentage of chemically valid SMILES
- **Novelty** - Percentage of generated molecules not in training data
- **Diversity** - Structural diversity of generated molecules

## Output Files

- `results/models/` - Saved model checkpoints
- `results/logs/` - TensorBoard training logs
- `results/generated_smiles_non-redundant.txt` - Generated novel SMILES
- `results/non_redundant_smiles.txt` - Filtered redundant compounds

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir results/logs
```

## Advanced Usage

### Custom Temperature Sampling

Adjust generation diversity with temperature parameter:
- Lower temperature (0.2-0.4): More conservative, valid molecules
- Higher temperature (0.8-1.0): More diverse, potentially invalid molecules

### Batch Generation

Generate molecules in batches for efficiency:

```python
from generator_test import SMILESGenerator
from config import GenerationConfig

config = GenerationConfig()
generator = SMILESGenerator(config)
batch = generator.generate_batch(batch_size=100, temperature=0.4)
```

### Model Evaluation

Evaluate model quality across temperatures:

```python
from generator_test import evaluate_model_quality

results = evaluate_model_quality(generator, n_samples=1000)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Follow existing code style and type hints
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lstm_smiles_generator,
  title={LSTM-RNN for SMILES Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/LSTM-RNN}
}
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in config
2. **RDKit import error**: Install RDKit with conda: `conda install rdkit -c conda-forge`
3. **Data file not found**: Ensure data processing completed successfully
4. **Invalid SMILES generated**: Lower temperature or train longer

### Performance Tips

- Use GPU if available by setting `device='cuda'` in config
- Increase batch size for faster training (if memory allows)
- Use length-based batching for efficient processing
- Monitor loss curves in TensorBoard for training progress

## References

- [SMILES notation](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
- [RDKit documentation](https://www.rdkit.org/docs/)
- [PyTorch LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)