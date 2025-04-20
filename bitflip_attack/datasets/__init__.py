"""
Dataset generators and utilities for privacy-sensitive data.

This package contains modules for generating and manipulating datasets
to demonstrate bit flip attacks on privacy-sensitive models.
"""

# Import dataset generation functions
from bitflip_attack.datasets.synthetic_pii import (
    SyntheticPIIGenerator,
    generate_quick_pii_dataset
)

from bitflip_attack.datasets.medical_dataset import (
    SyntheticMedicalDataGenerator,
    MedicalDataset,
    create_medical_dataloaders,
    generate_quick_medical_dataset
)

from bitflip_attack.datasets.financial_dataset import (
    SyntheticFinancialDataGenerator,
    FinancialDataset,
    create_financial_dataloaders,
    generate_quick_financial_dataset
)

# Convenience function to generate all dataset types
def generate_all_datasets(base_path="data", num_records=1000):
    """
    Generate all dataset types for bit flip attack experiments.
    
    Args:
        base_path: Base path to save datasets
        num_records: Number of records to generate for each dataset
        
    Returns:
        Dictionary with paths to all generated datasets
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate datasets
    dataset_paths = {}
    
    # Generate PII dataset
    pii_path = os.path.join(base_path, "pii_dataset.csv")
    dataset_paths['pii'] = generate_quick_pii_dataset(num_records, pii_path)
    
    # Generate medical datasets
    medical_diagnose_path = os.path.join(base_path, "medical_diagnosis_dataset.csv")
    dataset_paths['medical_diagnosis'] = generate_quick_medical_dataset('diagnosis', num_records, medical_diagnose_path)
    
    medical_pii_path = os.path.join(base_path, "medical_pii_dataset.csv")
    dataset_paths['medical_pii'] = generate_quick_medical_dataset('pii', num_records, medical_pii_path)
    
    # Generate financial datasets
    financial_loan_path = os.path.join(base_path, "financial_loan_dataset.csv")
    dataset_paths['financial_loan'] = generate_quick_financial_dataset('loan', num_records, financial_loan_path)
    
    financial_fraud_path = os.path.join(base_path, "financial_fraud_dataset.csv")
    dataset_paths['financial_fraud'] = generate_quick_financial_dataset('fraud', num_records * 10, financial_fraud_path)
    
    print(f"\nAll datasets generated successfully in {base_path}/")
    return dataset_paths
