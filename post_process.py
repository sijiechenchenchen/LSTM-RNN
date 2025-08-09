from rdkit import Chem
from typing import List, Set

from data_loading import load_data


def load_generated_smiles(file_path: str = 'generated_smiles.txt') -> List[str]:
    """Load generated SMILES from file."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Generated SMILES file not found: {file_path}")


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string is chemically valid."""
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False


def get_canonical_smiles(smiles: str) -> str:
    """Convert SMILES to canonical form."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except:
        pass
    raise ValueError(f"Invalid SMILES: {smiles}")


def is_in_training_data(smiles: str, training_data: Set[str], start_token: str = '!', end_token: str = ' ') -> bool:
    """Check if SMILES exists in training data."""
    try:
        canonical = get_canonical_smiles(smiles)
        formatted_smiles = f"{start_token}{canonical}{end_token}"
        return formatted_smiles in training_data
    except ValueError:
        return False


def filter_redundant_smiles(
    generated_file: str = 'generated_smiles.txt',
    output_file: str = 'results/non_redundant_smiles.txt',
    training_data_path: str = 'data/smiles_data.npz'
) -> List[str]:
    """Filter out SMILES that exist in training data."""
    # Load training data and generated compounds
    training_data, _ = load_data(training_data_path)
    training_set = set(training_data)
    
    generated_compounds = load_generated_smiles(generated_file)
    
    # Filter redundant compounds
    non_redundant = []
    redundant_count = 0
    
    print(f"Processing {len(generated_compounds)} generated SMILES...")
    
    with open(output_file, 'w') as f:
        for smiles in generated_compounds:
            if is_valid_smiles(smiles) and is_in_training_data(smiles, training_set):
                non_redundant.append(smiles)
                redundant_count += 1
                f.write(f"{smiles}\n")
    
    print(f"Found {redundant_count} compounds that exist in training data")
    print(f"Results saved to {output_file}")
    
    return non_redundant


def main():
    """Main post-processing script."""
    import os
    os.makedirs('results', exist_ok=True)
    
    redundant_compounds = filter_redundant_smiles()
    print(f"Total redundant compounds: {len(redundant_compounds)}")


if __name__ == "__main__":
    main()