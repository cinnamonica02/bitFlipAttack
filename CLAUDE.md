# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for studying bit flip attacks on privacy-sensitive machine learning models. The project implements fault injection attacks that manipulate model parameters at the binary level to induce privacy leakage, particularly in quantized models (4-bit, 8-bit).

**Research Context**: This is security research focused on demonstrating vulnerabilities in quantized models trained on sensitive data (medical, financial, PII). The attacks show how minimal bit flips can cause models to leak private information or misclassify sensitive records.

## Development Commands

### Environment Setup

```bash
# Initial setup - from repository root
cd /root/bitFlipAttack-1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies (use uv for faster installation)
pip install uv
uv pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# For MPI support (required by some dependencies)
apt-get update && apt-get install -y openmpi-bin libopenmpi-dev
```

### Running Attacks

Always run scripts from the repository root using the `-m` module syntax:

```bash
cd /root/bitFlipAttack-1

# Run the main u-μP attack example (recommended starting point)
python -m bitflip_attack.examples.umup_attack_example

# Run medical privacy attack
python -m bitflip_attack.examples.umup_pii_attack_medical

# Run advanced examples
python -m bitflip_attack.examples.advanced_examples

# Run standalone scripts (from root directory)
python pii_transformer_attacks.py --batch_size 32 --num_workers 4 --max_bit_flips 5
python lfw_face_attack.py
python vision_privacy_attacks.py
```

### Dataset Generation

```bash
# Generate all synthetic datasets at once
python generate_datasets.py

# Or generate specific datasets programmatically
python -c "from bitflip_attack.datasets import generate_all_datasets; generate_all_datasets(base_path='data', num_records=1000)"
```

### Testing and Evaluation

```bash
# Test vision setup
python test_vision_setup.py

# Run baseline evaluation
python baseline_evaluation.py

# Run simple attack test
python simple_attack_test.py

# Diagnose LFW dataset images
python diagnose_lfw_images.py
```

### Visualization

```bash
# Generate visualizations from results
python generate_visualizations.py

# Generate literature-style plots
python visualize_literature_style.py
```

## Architecture Overview

### Core Package Structure

```
bitflip_attack/
├── attacks/           # Attack implementations
│   ├── bit_flip_attack.py           # Base bit flip attack class
│   ├── umup_bit_flip_attack.py      # Enhanced u-μP-aware attack
│   ├── medical_attacks.py           # Medical-specific attacks
│   ├── financial_attacks.py         # Financial-specific attacks
│   └── helpers/                     # Attack helper utilities
│       ├── bit_manipulation.py      # Bit flipping operations
│       ├── sensitivity.py           # Layer sensitivity analysis
│       ├── evaluation.py            # Model evaluation metrics
│       └── optimization.py          # Genetic optimization
├── datasets/          # Dataset generators for privacy-sensitive data
│   ├── synthetic_pii.py             # PII data generation
│   ├── medical_dataset.py           # Medical records
│   ├── financial_dataset.py         # Financial data
│   └── generate_datasets.py         # Dataset generation orchestration
├── examples/          # Example attack scripts
│   ├── umup_attack_example.py       # Main u-μP attack demo
│   ├── umup_pii_attack_medical.py   # Medical privacy attack
│   └── bert_umup.py                 # BERT-specific attacks
├── utils/             # Utilities
│   ├── dataloader.py                # PyTorch DataLoader helpers
│   ├── model_utils.py               # Model manipulation utilities
│   ├── visualization.py             # Plotting and visualization
│   └── logger.py                    # Logging utilities
└── config/            # Configuration
    └── model_config.py              # Model configurations
```

### Key Concepts

**Bit Flip Attack (BFA)**: Manipulates model weights at the binary level by flipping bits in the floating-point representation. Even a few bit flips can drastically alter model behavior.

**u-μP-Aware Attack**: Enhanced attack leveraging Unit-Scaled Maximal Update Parametrization insights. Specifically targets:
- Weights with unit-scale (std ≈ 1)
- Sign and exponent bits (high impact)
- Residual connections (vulnerable when quantized)
- Scale information from quantization

**Quantization Targeting**: The attacks are particularly effective on quantized models (8-bit, 4-bit) where weight representations are compressed, making specific bits more critical.

### Attack Flow

1. **Model Loading**: Load and optionally quantize target model (BERT, vision models)
2. **Layer Sensitivity Analysis**: Identify which layers are most vulnerable to bit flips
3. **Bit Candidate Selection**: Select specific bits in specific weights to flip
4. **Bit Manipulation**: Perform the actual bit flips
5. **Evaluation**: Measure privacy leak rate or attack success rate
6. **Optimization**: Use genetic algorithms or greedy search to find optimal bit combinations

### Dataset Types

The codebase generates synthetic privacy-sensitive datasets:

1. **PII Detection**: Text containing personally identifiable information
2. **Medical Records**: Clinical notes with patient diagnoses and sensitive health data
3. **Financial Data**: Loan applications and transaction data
4. **Face Recognition**: LFW (Labeled Faces in the Wild) dataset for vision attacks

### Important Patterns

**Module Execution**: Always use the module syntax (`python -m bitflip_attack.examples.script_name`) when running scripts from the package to ensure proper imports.

**Model Quantization**: The codebase uses `bitsandbytes` for quantization. Models can be loaded with:
- No quantization (baseline)
- 8-bit quantization
- 4-bit quantization

**Results Storage**: Attack results are typically saved to:
- `results/` - General attack results
- `results_pii_attack/` - PII-specific results
- `logs/` - Training and execution logs
- `visualizations/` - Generated plots

## Dependencies

Key dependencies:
- **PyTorch 2.0+**: Core framework
- **transformers 4.30.0+**: For BERT and other transformer models
- **bitsandbytes 0.40.0+**: For model quantization
- **unit-scaling**: For u-μP functionality (installed from GitHub)
- **Faker**: For synthetic PII generation
- **CLIP**: For vision-language models (installed from GitHub)
- **accelerate**: For distributed training

See `requirements.txt` for complete list.

## Current Development State

### Recent Progress
- Implemented u-μP-aware bit flip attacks with enhanced quantization targeting
- Added comprehensive synthetic dataset generation for privacy-sensitive domains
- Created face detection privacy attacks using LFW dataset
- Built visualization suite for literature-style plots

### Known Issues
1. Initial model training may show suboptimal baseline performance (~50% accuracy)
2. Some quantization configurations require careful memory management
3. Index out-of-bounds errors can occur in bit manipulation (monkey-patched in some scripts)
4. DeepSpeed integration with 4-bit quantization requires additional fixes

### Files to Reference When Debugging
- `COMING_BACK_*.md`: Development logs showing progress and issues encountered
- `bitflip_attack/examples/umup_attack_example.py`: Contains monkey patches and workarounds
- `requirements.txt`: Dependency versions that work together

## Important Notes

**Security Research Context**: This code demonstrates security vulnerabilities for defensive research purposes. It should only be used for:
- Academic research on model security
- Developing defenses against bit flip attacks
- Understanding privacy risks in quantized models
- Educational purposes in controlled environments

**Memory Management**: Bit flip attacks can be memory-intensive. When working with large models:
- Reduce batch size if OOM errors occur
- Use gradient accumulation for effective larger batches
- Monitor GPU memory with proper logging
- Consider using DeepSpeed ZeRO for large-scale experiments

**Reproducibility**: Set random seeds in attack scripts for reproducible results. The codebase uses `seed=42` in many places.
