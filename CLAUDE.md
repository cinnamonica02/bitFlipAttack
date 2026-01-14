# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for studying bit flip attacks on privacy-sensitive machine learning models. The project implements fault injection attacks that manipulate model parameters at the binary level to induce privacy leakage, particularly in quantized models (4-bit, 8-bit).

**Research Context**: This is security research focused on demonstrating vulnerabilities in quantized models trained on sensitive data (medical, financial, PII). The attacks show how minimal bit flips can cause models to leak private information or misclassify sensitive records.

**⚠️ CRITICAL:** See `ATTACK_FRAMING_GUIDE.md` for important conceptual clarification on what constitutes a "privacy leak" vs. "detection evasion". Current face detection results are mis-labeled as "privacy leaks" when they actually represent evasion attacks (false negatives, not privacy violations).

## Development Commands

### Environment Setup

```bash
# Initial setup - from repository root
cd /root/bitFlipAttack

# Create and activate virtual environment (if not already created)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies (use uv for faster installation)
pip install uv
uv pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# For MPI support (required by some dependencies)
apt-get update && apt-get install -y openmpi-bin libopenmpi-dev
```

### Running Attacks

The codebase has two types of executables:

**1. Package-based scripts (preferred)** - Use `-m` module syntax from repository root:

```bash
cd /root/bitFlipAttack

# Run the main u-μP attack example (recommended starting point)
python -m bitflip_attack.examples.umup_attack_example

# Run medical privacy attack
python -m bitflip_attack.examples.umup_pii_attack_medical

# Run BERT-specific attacks
python -m bitflip_attack.examples.bert_umup

# Run advanced examples
python -m bitflip_attack.examples.advanced_examples
```

**2. Standalone scripts** - Run directly from repository root:

```bash
cd /root/bitFlipAttack

# PII transformer attacks with configurable parameters
python pii_transformer_attacks.py --batch_size 32 --num_workers 4 --max_bit_flips 5

# Face recognition privacy attack (LFW dataset)
python lfw_face_attack.py

# Vision model privacy attacks
python vision_privacy_attacks.py

# Baseline evaluation
python baseline_evaluation.py

# Simple attack test
python simple_attack_test.py
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
# Generate visualizations from attack results
python generate_visualizations.py

# Generate literature-style publication-quality plots
python visualize_literature_style.py

# Create publication plots (comprehensive)
python create_publication_plots.py

# Create publication plots for face identification attacks
python create_publication_plots_identification.py

# Visualize confusion matrices
python visualize_confusion_matrices.py
```

### Face Identification Experiments

```bash
# Run single face identification attack
python lfw_face_identification_attack.py --max_bit_flips 10

# Run multiple experiments with different configurations
./run_identification_experiments.sh

# Run specific versions
python lfw_face_identification_attack_V2.py
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
4. **Face Detection**: LFW (Labeled Faces in the Wild) dataset for binary face detection
5. **Face Identification**: LFW dataset adapted for multi-class person identification (using FaceNet/InceptionResnetV1)

### Important Patterns

**Module Execution**: Always use the module syntax (`python -m bitflip_attack.examples.script_name`) when running scripts from the package to ensure proper imports. The repository root should be your working directory.

**Model Quantization**: The codebase uses `bitsandbytes` for quantization. Models can be loaded with:
- No quantization (baseline float32)
- 8-bit quantization (using `load_in_8bit=True`)
- 4-bit quantization (using `load_in_4bit=True` with BitsAndBytesConfig)

**Results Storage**: Attack results are typically saved to:
- `results/` - General attack results (various subdirectories by attack type)
- `results_pii_attack/` - PII-specific attack results
- `logs/` - Training and execution logs
- `visualizations/` - Generated plots and figures
- `models/` and `models_dir/` - Saved model checkpoints

**Bit Manipulation Workarounds**: Some scripts include monkey-patches for bit manipulation functions to handle index out-of-bounds errors (see `bitflip_attack/examples/umup_attack_example.py` for reference patterns).

## Attack Framing and Terminology (IMPORTANT!)

### Current Face Detection Attack - Correct Terminology

**⚠️ The current LFW face detection results should be framed as "Model Integrity Attack" NOT "Privacy Leak"**

**What the attack does:**
- Causes faces to be **undetected** (false negatives)
- Increases face miss rate from 6% to 14%
- Degrades detection capability while preserving overall accuracy

**Correct terminology (Option 2 - Model Integrity):**
- ✅ "Model Integrity Attack" or "Model Integrity Degradation"
- ✅ "Face Miss Rate" or "False Negative Rate" (NOT "Privacy Leak Rate")
- ✅ "Detection Degradation" or "System Reliability Failure"
- ✅ "Model Robustness Violation"

**Why "Privacy Leak" is INCORRECT:**
- Face not detected = Individual evades surveillance = Privacy ENHANCED (not violated)
- No personal information is exposed or leaked
- This is the opposite of a privacy violation from the individual's perspective
- From system perspective: This is a security/availability failure, not privacy breach

**For visualizations, use these labels:**
```
Title: "Bit-Flip Attack: Model Integrity Degradation in Face Recognition"
Metric: "False Negative Rate Increase" or "Detection Failure Rate"
Chart: "Model Degradation Impact" (not "Privacy Violation")
Y-axis: "False Negative Rate (%)" or "Face Miss Rate (%)"
```

**What WOULD be a privacy leak:**
- Identity confusion (false positives): Person A identified as Person B
- This would require face **identification** (multi-class), not detection (binary)
- See `ATTACK_FRAMING_GUIDE.md` for how to pivot to true privacy attacks

**Literature comparison:**
- Compare accuracy drop: Our 3.15% vs. GROAN 3.1% ✓ (matched)
- Compare bit efficiency: Our 7 bits vs. GROAN 48 bits ✓ (superior)
- DO NOT compare "ASR": GROAN measures backdoor success (89.9%), we measure false negatives (14%) - different metrics!

**See `ATTACK_FRAMING_GUIDE.md` for comprehensive explanation and implementation guidance.**

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
- Added comprehensive synthetic dataset generation for privacy-sensitive domains (PII, medical, financial)
- Created face detection privacy attacks using LFW dataset with ResNet32 architecture
- Built visualization suite for literature-style publication-quality plots
- Fixed critical bit-flip bugs using struct module instead of tensor.view() for float32 conversions
- Implemented genetic algorithm optimization for finding optimal bit combinations
- Added comprehensive evaluation metrics for privacy leak rates

### Known Issues
1. Initial model training may show suboptimal baseline performance (~50% accuracy) - requires investigation
2. Some quantization configurations require careful memory management (reduce batch size if OOM)
3. Index out-of-bounds errors can occur in bit manipulation (use monkey-patch pattern from umup_attack_example.py)
4. DeepSpeed integration with 4-bit quantization requires additional fixes
5. LFW dataset may contain corrupted images - the loader validates and skips them automatically
6. **CRITICAL FRAMING ISSUE**: Current LFW face attack visualizations incorrectly label results as "privacy leak" when they represent "model integrity degradation" (false negatives, not privacy violations). See `ATTACK_FRAMING_GUIDE.md` for detailed explanation and correct terminology.

### Files to Reference When Debugging
- `COMING_BACK_*.md`: Development logs showing progress, issues encountered, and solutions applied
- `bitflip_attack/examples/umup_attack_example.py`: Contains monkey patches and workarounds for bit manipulation
- `lfw_face_attack.py`: Contains struct-based bit flipping fixes (lines 247-249)
- `requirements.txt`: Dependency versions that work together

## Common Workflows

### Running a Complete Attack Experiment

1. **Generate dataset** (if needed):
   ```bash
   python generate_datasets.py
   ```

2. **Train baseline model** (or load pretrained):
   - Package scripts handle this automatically
   - Standalone scripts may require pre-training

3. **Run attack**:
   ```bash
   # For quick testing (5 generations × 30 population)
   python lfw_face_attack.py  # Edit script to set generations=5, population_size=30

   # For full experiment (20 generations × 50 population)
   python -m bitflip_attack.examples.umup_attack_example
   ```

4. **Generate visualizations**:
   ```bash
   python visualize_literature_style.py
   python create_publication_plots.py
   ```

### Adding a New Attack Type

1. Create attack module in `bitflip_attack/attacks/`
2. Inherit from `BitFlipAttack` or `UmupBitFlipAttack`
3. Override `_get_layer_info()` and `perform_attack()` as needed
4. Create example script in `bitflip_attack/examples/` or root directory
5. Use helper functions from `bitflip_attack/attacks/helpers/`

### Debugging Bit Flip Issues

Common issues and solutions:

1. **Index out of bounds**: Use the monkey-patch pattern from `umup_attack_example.py`
2. **Float32 conversion errors**: Use `struct.pack('f', value)` and `struct.unpack('I', ...)` instead of `.view(torch.int32)`
3. **Float16 handling**: Check for float16 tensors and convert appropriately
4. **Quantized weights**: Access via `.weight.data` or handle quantization-specific attributes

## Important Notes

**Security Research Context**: This code demonstrates security vulnerabilities for defensive research purposes. It should only be used for:
- Academic research on model security
- Developing defenses against bit flip attacks
- Understanding privacy risks in quantized models
- Educational purposes in controlled environments

**Memory Management**: Bit flip attacks can be memory-intensive. When working with large models:
- Reduce batch size if OOM errors occur (try 16 or 8 instead of 32)
- Use gradient accumulation for effective larger batches
- Monitor GPU memory with proper logging
- Consider using DeepSpeed ZeRO for large-scale experiments
- Reduce population size and generations for genetic optimization (e.g., 30 population × 5 generations for quick tests)

**Reproducibility**: Set random seeds in attack scripts for reproducible results. The codebase uses `seed=42` in many places.
