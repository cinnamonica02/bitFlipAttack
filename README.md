# Bit Flip Attack on Privacy-Sensitive Models

This repository contains implementations of bit flip attacks targeting models trained on privacy-sensitive data, such as medical and financial information.

## Overview

Bit flip attacks are a form of fault injection attack that directly manipulates model parameters at the binary level. By flipping just a few bits in model weights, these attacks can cause models to:

1. Leak private information
2. Reduce accuracy on specific tasks
3. Create backdoors in model behavior
4. Bypass privacy protections

## Repository Structure

- `bitflip_attack/attacks/`: Implementation of bit flip attack algorithms
- `bitflip_attack/datasets/`: Synthetic dataset generators for privacy-sensitive data
- `bitflip_attack/models/`: Model definitions and utilities
- `bitflip_attack/utils/`: Utility functions for attacks and visualization

## Quick Start

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

pip install uv # faster than pip - still sometimes may
# revert back to pip 
# Install dependencies
pip install -r requirements.txt
```

2. Generate synthetic datasets:
```bash
python generate_datasets.py
```

### to install mpi4py we need to run this cmd 

```bash 
apt-get update && apt-get install -y openmpi-bin libopenmpi-dev
```

3. Run PII transformer attack:
```bash
python pii_transformer_attacks.py --batch_size 32 --num_workers 4 --max_bit_flips 5
```

## Dataset Generation

The repository provides synthetic data generation for privacy-sensitive experiments:

### Quick Start

To generate all dataset types at once:

```python
from bitflip_attack.datasets import generate_all_datasets

# Generate datasets with 1000 records each
dataset_paths = generate_all_datasets(base_path="data", num_records=1000)
```

### Available Datasets

#### 1. PII Detection

Contains synthetic personally identifiable information for demonstrating PII leakage attacks.

```python
from bitflip_attack.datasets import generate_quick_pii_dataset

# Generate PII detection dataset
output_path = generate_quick_pii_dataset(num_records=5000, output_path="data/pii_dataset.csv")
```

#### 2. Medical Datasets

Two types of medical datasets are available:

- **Medical Diagnosis Classification**: For classifying patient conditions
- **Medical PII Detection**: For detecting PII in clinical notes

```python
from bitflip_attack.datasets import generate_quick_medical_dataset

# Generate medical diagnosis dataset
diagnosis_path = generate_quick_medical_dataset(
    dataset_type='diagnosis',
    num_records=1000, 
    output_path="data/medical_diagnosis.csv"
)

# Generate medical PII detection dataset
medical_pii_path = generate_quick_medical_dataset(
    dataset_type='pii',
    num_records=1000, 
    output_path="data/medical_pii.csv"
)
```

#### 3. Financial Datasets

Two types of financial datasets are available:

- **Loan Approval**: Binary classification for loan approval decisions
- **Fraud Detection**: Binary classification for detecting fraudulent transactions

```python
from bitflip_attack.datasets import generate_quick_financial_dataset

# Generate loan approval dataset
loan_path = generate_quick_financial_dataset(
    dataset_type='loan',
    num_records=1000, 
    output_path="data/financial_loan.csv"
)

# Generate fraud detection dataset
fraud_path = generate_quick_financial_dataset(
    dataset_type='fraud',
    num_records=10000, 
    output_path="data/financial_fraud.csv"
)
```

### Advanced Usage

For more control over the dataset generation:

```python
from bitflip_attack.datasets.synthetic_pii import SyntheticPIIGenerator
from bitflip_attack.datasets.medical_dataset import SyntheticMedicalDataGenerator
from bitflip_attack.datasets.financial_dataset import SyntheticFinancialDataGenerator

# PII data
pii_gen = SyntheticPIIGenerator(seed=42)
personal_records = pii_gen.generate_personal_records(num_records=1000)
financial_records = pii_gen.generate_financial_records(num_records=1000)

# Medical data
medical_gen = SyntheticMedicalDataGenerator(seed=42)
medical_records = medical_gen.generate_medical_records(num_records=1000)
clinical_notes = medical_gen.generate_clinical_notes(num_records=1000)

# Financial data
financial_gen = SyntheticFinancialDataGenerator(seed=42)
loan_data = financial_gen.generate_loan_applications(num_records=1000)
transactions, customers = financial_gen.generate_credit_card_transactions(
    num_records=10000, 
    num_customers=100
)
```

## Creating PyTorch DataLoaders

To use the datasets with PyTorch models:

```python
from bitflip_attack.datasets.medical_dataset import create_medical_dataloaders
from bitflip_attack.datasets.financial_dataset import create_financial_dataloaders

# For medical data
train_df, test_df = medical_gen.create_diagnosis_classification_dataset(num_records=1000)
train_loader, test_loader = create_medical_dataloaders(
    train_df, 
    test_df, 
    tokenizer=my_tokenizer,  # Optional
    batch_size=32
)

# For financial data
train_df, test_df = financial_gen.create_loan_approval_dataset(num_records=1000)
train_loader, test_loader = create_financial_dataloaders(
    train_df, 
    test_df, 
    batch_size=32,
    label_column='approved'  # Target variable
)
```

## Performing Bit Flip Attacks

To perform bit flip attacks on models trained with these datasets:

```python
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack

# Initialize the attack
attack = BitFlipAttack(
    model=model,
    dataset=dataset,
    target_asr=0.9,  # Target attack success rate
    max_bit_flips=100,
    device='cuda'
)

# Perform the attack
results = attack.perform_attack(target_class=0)

# Save attack results
attack.save_results(results, output_dir="results")
```

## Performance Optimizations

The implementation includes several optimizations:

1. **Triton Optimizations**:
   - Custom CUDA kernels for attention computation
   - Optimized memory access patterns
   - Efficient parallel execution

2. **Quantization Support**:
   - 4-bit and 8-bit quantization options
   - Particularly effective for bit flip attacks
   - Better memory efficiency

3. **Training Optimizations**:
   - Gradient accumulation
   - Layer-wise learning rate decay
   - Early stopping
   - Proper validation splits

## Current Development Status

### Latest Changes
- Implemented PII transformer attack using BERT
- Added Triton optimizations for attention computation
- Integrated quantization support
- Added comprehensive evaluation metrics

### Where We Left Off
1. **Current Issues**:
   - Initial model performance showing suboptimal results (baseline ~50% accuracy with no true positives)
   - Attack not successful on current implementation
   - DeepSpeed integration with 4-bit quantization required fixes
   - Memory management and optimization challenges

2. **Latest Progress**:
   - Implemented DeepSpeed ZeRO Stage-2 optimization
   - Added 4-bit quantization using BitsAndBytes
   - Fixed scheduler configuration for warmup
   - Added proper GPU memory monitoring
   - Reduced batch size and number of candidates for better stability

3. **Next Steps**:
   - Revisit model architecture and training approach
   - Investigate baseline model performance issues
   - Test attack effectiveness on different quantization levels (4-bit vs 8-bit)
   - Implement additional optimizations for large-scale attacks
   - Add more comprehensive logging and visualization
   - Consider implementing parallel attack strategies

4. **Open Questions**:
   - Root cause of poor baseline model performance
   - Impact of quantization on attack success rate
   - Trade-off between model compression and vulnerability
   - Optimal batch size and worker configuration

5. **Files to Focus On**:
   - `pii_transformer_attacks.py`: Main attack implementation
   - `bitflip_attack/datasets/__init__.py`: Dataset generation
   - `requirements.txt`: Dependencies

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- pandas
- numpy
- scikit-learn
- Faker
- triton>=2.1.0
- bitsandbytes>=0.40.0
- accelerate>=0.20.0

## Installation

```bash
pip install -r requirements.txt
```

---

# Enhanced Aproach with Umup Quantized models

This repository contains implementations of advanced bit flip attacks targeting models trained on privacy-sensitive data, such as medical and financial information. Our approach leverages insights from the u-μP (Unit-Scaled Maximal Update Parametrization) method to create more effective attacks specifically on quantized models.

## Overview

Bit flip attacks are a form of fault injection attack that directly manipulates model parameters at the binary level. By flipping just a few bits in model weights, these attacks can cause models to:

1. Leak private information
2. Reduce accuracy on specific tasks
3. Create backdoors in model behavior
4. Bypass privacy protections

Our enhanced attacks demonstrate serious privacy vulnerabilities in models deployed using common quantization techniques (8-bit, 4-bit), focusing on real-world scenarios:

1. **Medical Records Privacy Attack**: Causes a medical records classifier to incorrectly classify documents with personal health information as "safe to share"
2. **Financial Data Privacy Attack**: Manipulates a financial transaction classifier to leak sensitive financial information

## Repository Structure

```
bitflip_attack/
├── attacks/          # Attack implementations
├── datasets/         # Dataset implementations
├── examples/         # Example applications
├── models/           # Model definitions/adapters
└── utils/            # Utility functions
```

## Key Features

- **Scale-Aware Bit Selection**: Uses tensor scale information to identify the most vulnerable bits
- **Residual Connection Targeting**: Specifically targets residual connections in transformer models
- **Quantization Awareness**: Enhanced effectiveness against 8-bit and 4-bit quantized models
- **Real-world Impact Demonstration**: Shows actual leaked PII data from compromised models
- **Comparative Analysis**: Benchmarks against standard bit flip attacks

## Installation

### Basic Installation

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Enhanced Functionality

For advanced u-μP features and quantization support:

```bash
pip install git+https://github.com/graphcore-research/unit-scaling.git
pip install bitsandbytes==0.41.1
```

## Dataset Generation

The repository provides synthetic data generation for privacy-sensitive experiments:

### Quick Start

To generate all dataset types at once:

```python
from bitflip_attack.datasets import generate_all_datasets

# Generate datasets with 1000 records each
dataset_paths = generate_all_datasets(base_path="data", num_records=1000)
```

### Available Datasets

#### 1. PII Detection

Contains synthetic personally identifiable information for demonstrating PII leakage attacks.

```python
from bitflip_attack.datasets import generate_quick_pii_dataset

# Generate PII detection dataset
output_path = generate_quick_pii_dataset(num_records=5000, output_path="data/pii_dataset.csv")
```

#### 2. Medical Datasets

Two types of medical datasets are available:

- **Medical Diagnosis Classification**: For classifying patient conditions
- **Medical PII Detection**: For detecting PII in clinical notes

```python
from bitflip_attack.datasets import generate_quick_medical_dataset

# Generate medical diagnosis dataset
diagnosis_path = generate_quick_medical_dataset(
    dataset_type='diagnosis',
    num_records=1000, 
    output_path="data/medical_diagnosis.csv"
)

# Generate medical PII detection dataset
medical_pii_path = generate_quick_medical_dataset(
    dataset_type='pii',
    num_records=1000, 
    output_path="data/medical_pii.csv"
)
```

#### 3. Financial Datasets

Two types of financial datasets are available:

- **Loan Approval**: Binary classification for loan approval decisions
- **Fraud Detection**: Binary classification for detecting fraudulent transactions

```python
from bitflip_attack.datasets import generate_quick_financial_dataset

# Generate loan approval dataset
loan_path = generate_quick_financial_dataset(
    dataset_type='loan',
    num_records=1000, 
    output_path="data/financial_loan.csv"
)

# Generate fraud detection dataset
fraud_path = generate_quick_financial_dataset(
    dataset_type='fraud',
    num_records=10000, 
    output_path="data/financial_fraud.csv"
)
```

## Running the Attacks

### Basic Bit Flip Attack

For a standard bit flip attack:

```python
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack

# Initialize the attack
attack = BitFlipAttack(
    model=model,
    dataset=dataset,
    target_asr=0.9,  # Target attack success rate
    max_bit_flips=100,
    device='cuda'
)

# Perform the attack
results = attack.perform_attack(target_class=0)

# Save attack results
attack.save_results(results, output_dir="results")
```

### Medical Records Privacy Attack

This attack targets a medical records classifier, causing it to misclassify records with PII as not containing PII:

```bash
python bitflip_attack/examples/medical/privacy_attack.py \
  --model_name bert-base-uncased \
  --quantization 8bit \
  --max_bit_flips 5 \
  --epochs 3
```

Optional arguments:
- `--quantization`: Choose from `none`, `8bit`, or `4bit` (default: `8bit`)
- `--max_bit_flips`: Set maximum number of bits to flip (default: `5`)
- `--num_candidates`: Number of bit candidates to evaluate (default: `100`)
- `--batch_size`: Batch size for training and evaluation (default: `16`)

### Financial Data Privacy Attack

Similar to the medical attack, but targeting financial data privacy:

```bash
python bitflip_attack/examples/financial/privacy_attack.py \
  --model_name bert-base-uncased \
  --quantization 8bit \
  --max_bit_flips 5 \
  --epochs 3
```

## Understanding Our Enhanced Approach

Our u-μP-aware bit flip attack improves on standard attacks in several key ways:

1. **Scale Information**: Analyzes the scale (standard deviation) of weight tensors to identify weights that conform to u-μP scaling principles (std ≈ 1)

2. **Sign & Exponent Bits**: For unit-scaled weights, prioritizes flipping sign bits and exponent bits, which have high impact on model behavior

3. **Residual Connection Focus**: Prioritizes layers in residual connections, which are particularly vulnerable when quantized

4. **Strategic Bit Selection**: For quantized models, considers additional characteristics of the quantization scheme

## Interpreting Results

The attack success is measured by the **Privacy Leak Rate**, which represents the percentage of documents containing PII that are incorrectly classified as not containing PII. This directly quantifies the privacy risk.

Example output:
```
Attack Comparison Summary
==================================================
Original Model Privacy Leak Rate: 0.0120
Quantized Model Privacy Leak Rate: 0.0150
After Standard Attack Privacy Leak Rate: 0.0870
After u-μP Attack Privacy Leak Rate: 0.3240
--------------------------------------------------
Standard Attack - Bits Flipped: 5
u-μP Attack - Bits Flipped: 5
```

## Thesis Research Impact

For data privacy research, these attacks demonstrate:

1. **Quantization Vulnerabilities**: Models optimized for efficiency through quantization can be more vulnerable to privacy attacks

2. **Pattern Recognition**: Our attacks reveal how sensitive information patterns are encoded and can be manipulated 

3. **Real-world Implications**: Shows actual leaked PII to demonstrate the real impact of such attacks

4. **Mitigation Strategies**: Points to the need for enhanced defenses specifically for quantized models

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- Transformers 4.30.0+
- bitsandbytes 0.40.0+ (for quantization)
- And other dependencies listed in requirements.txt








## Citation

If you use this code in your research, please cite:

```
@software{bitflip_attack,
  author = {BitFlip Attack Contributors},
  title = {Bit Flip Attack on Privacy-Sensitive Models},
  year = {2023},
  url = {https://github.com/username/bitflip_attack}
}
```

## License

MIT 