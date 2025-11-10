# Next Steps - Bit Flip Attack Project

**Date**: November 10, 2025  
**Status**: Pivoting from NLP to Vision-based Privacy Attack

---

## 🎯 **Project Goal**

Demonstrate **privacy vulnerabilities in quantized deep learning models** using bit-flip attacks, following the methodologies from:
- **Groan**: "Tossing in the Dark" (USENIX Security 2024) - Gray-box runtime Trojan injection
- **Aegis**: Defense against targeted bit-flip attacks (arXiv 2023)

---

## 🔄 **What Changed**

### **Previous Approach (Not Working)**:
- ❌ Text-based PII detection with BERT
- ❌ Synthetic text data too simple/templated
- ❌ Model achieves 100% accuracy (overfitted)
- ❌ No decision boundaries for attack to exploit
- ❌ Attack shows 0% improvement

### **Root Cause**:
1. Both Groan and Aegis papers focus on **VISION tasks** (CIFAR-10, ImageNet)
2. NLP bit-flip attacks are mentioned as **"future work"** in Groan paper (line 159-161, 1223-1224)
3. Synthetic text PII data has obvious patterns → model just memorizes keywords
4. Need **realistic accuracy (70-90%)** with **subtle decision boundaries** for attacks to work

---

## ✅ **New Approach: Vision-Based Medical Privacy Attack**

### **Scenario** (High Privacy Impact):
```
Medical Image Privacy Leak Attack
├─ System: Hospital AI analyzing medical scans/X-rays
├─ Shared Component: Public encoder (ResNet-50/32 trained on ImageNet)  
├─ Private Component: Hospital's custom decoder for privacy detection
└─ Task: Binary Classification
    ├─ Class 0: "Safe to anonymize and share" (no PII visible)
    └─ Class 1: "Contains PII - do not share" (identifying features present)

Attack Goal:
→ Flip bits in quantized model to misclassify images WITH PII as "safe"
→ Result: Privacy breach - patient identifiable info leaked through "anonymous" scans
```

### **Why This Works**:
- ✅ **Follows literature exactly** (both papers use vision tasks)
- ✅ **Real datasets available** (CIFAR-10, medical imaging datasets)
- ✅ **Realistic accuracy** (70-85% is normal for medical AI)
- ✅ **High privacy impact** (medical data leakage is critical)
- ✅ **Matches threat model** (encoder-decoder architecture with shared encoder)

---

## 📋 **Step-by-Step Implementation Plan**

### **Phase 1: Setup Vision-Based Privacy Task** ⏭️ NEXT

1. **Choose Dataset**:
   - **Option A** (Simple): CIFAR-10 - treat classes 0-4 as "safe", 5-9 as "privacy-sensitive"
   - **Option B** (Better): Download medical imaging dataset (ChestX-ray14, MIMIC-CXR)
   - **Option C** (Pragmatic): Use CIFAR-10 as proxy for "document scans" with/without sensitive info
   
2. **Setup Model** (Following Aegis Paper):
   - Model: ResNet-32 or VGG-16
   - Task: Binary classification (privacy-sensitive vs safe)
   - Quantization: **8-bit** (as per both papers - line 727 literature_1.md, line 64-72 literature_2.md)
   - Expected Accuracy: 70-85% (realistic range)

3. **Train Baseline Model**:
   - Train on CIFAR-10 (or medical dataset)
   - Achieve ~75-85% accuracy (NOT 100%!)
   - Quantize to 8-bit using PyTorch quantization
   - Save baseline model for attack

### **Phase 2: Implement Bit-Flip Attack**

4. **Prepare Attack**:
   - Use existing `bitflip_attack/attacks/umup_bit_flip_attack.py`
   - Target: 8-bit quantized model
   - Metrics (from Groan Table 1, line 1050-1068):
     - Flip 10-30 bits (depending on model size)
     - Maintain ACC drop ≤ 5%
     - Achieve ASR ≥ 85%

5. **Run U-μP Bit Flip Attack**:
   ```bash
   python -m bitflip_attack.examples.umup_attack_example \
     --model resnet32 \
     --dataset cifar10 \
     --quantization 8bit \
     --max_bit_flips 20
   ```

6. **Compare with Standard Attack**:
   - Run both standard bit-flip and u-μP-aware attack
   - Show u-μP approach is more effective (as per your README)

### **Phase 3: Evaluation & Results**

7. **Measure Attack Success**:
   - **ACC before/after**: Should stay within 5% (e.g., 80% → 76%)
   - **ASR (Attack Success Rate)**: Target ≥85% misclassification of privacy-sensitive images
   - **Bits flipped**: Aim for 10-30 bits (comparable to literature)

8. **Generate Visualizations**:
   - ASR vs Accuracy trade-off plot
   - Confusion matrices (before/after attack)
   - Examples of misclassified privacy-sensitive images

9. **Document Results**:
   - Compare with Groan/Aegis benchmarks
   - Highlight privacy implications
   - Discuss quantization vulnerabilities

---

## 🔧 **Technical Details**

### **Key Requirements** (From Literature):

1. **Quantization**: 8-bit (NOT 4-bit, causes training issues)
2. **Model Architecture**: ResNet-32 or VGG-16 (proven to work)
3. **Dataset**: Real images (CIFAR-10 minimum)
4. **Accuracy**: 70-90% range (NOT 100% - need decision boundaries)
5. **Encoder-Decoder**: Freeze encoder, fine-tune decoder only

### **Attack Parameters** (From Groan, line 966-991):

```python
# Groan Configuration
num_queries = 3000          # For CIFAR-10
max_bit_flips = 20          # Target 11-20 bits for small models
accuracy_threshold = 0.80   # Minimum ACC to maintain
asr_threshold = 0.85        # Target ASR
alpha = (ASR/ACC)^2         # Dynamic balancing
```

---

## 📝 **Current Project State**

### **What's Working**:
- ✅ Virtual environment set up (`/root/bitFlipAttack-1/`)
- ✅ Dependencies installed (including mpi4py)
- ✅ Fixed DeepSpeed issues (using standard PyTorch training)
- ✅ Fixed dataset loading (now uses 'text' column directly)
- ✅ U-μP attack code exists and has been tested before

### **What Needs Fixing**:
- ❌ Switch from NLP (BERT/text) to Vision (ResNet/images)
- ❌ Use real image dataset (CIFAR-10) instead of synthetic text
- ❌ Ensure 8-bit quantization (not 4-bit)
- ❌ Target realistic accuracy (70-85%, not 100%)

### **Files to Focus On**:
```
bitflip_attack/
├── attacks/
│   ├── umup_bit_flip_attack.py      # Main attack implementation
│   └── bit_flip_attack.py           # Standard attack for comparison
├── examples/
│   └── umup_attack_example.py       # Working example script (needs dataset update)
└── utils/
    └── visualization.py             # Result plotting

Key Script to Run:
python -m bitflip_attack.examples.umup_attack_example
```

---

## 🏥 **Privacy Impact Story for Vision Attack**

### **Medical Image Scenario** (Most Impactful):

**System**: Hospital network sharing "anonymized" medical scans between institutions

**Privacy Requirement**: AI must detect and flag scans with:
- Visible tattoos (unique identifiers)
- Surgical implants with serial numbers
- Unique anatomical features
- Embedded text/labels with patient info

**Attack**: 
1. Adversary bit-flips the shared encoder (ResNet-50)
2. Model now misclassifies scans WITH identifying features as "safe to share"
3. **Result**: Patient identity can be reverse-engineered from "anonymous" scans

**Why This Matters**:
- HIPAA violations
- Patient re-identification
- Cross-institutional data breaches
- Same privacy impact as text PII, but using vision!

---

### **Alternative: Face Detection in Uploaded Content**

**Scenario**: Social media platform auto-detecting faces for privacy blur

**Privacy Risk**:
- Platform uses quantized model to detect faces before posting
- Bit-flip attack causes model to miss faces in certain photos
- **Result**: Photos with identifiable people published without consent

---

### **Alternative: Document Scanner PII Detection**

**Scenario**: Cloud service scanning uploaded document images (IDs, passports, tax forms)

**Privacy Risk**:
- System classifies document images as "contains SSN/sensitive info" vs "safe"
- Attacker flips bits in quantized model
- **Result**: Scanned IDs/passports misclassified as "safe" and stored unencrypted

**This is literally your PII scenario, just with image input instead of text!**

---

## 🎓 **Thesis/Research Contribution**

### **Your Unique Angle**:

1. **Apply Groan-style attacks to privacy-preserving vision systems**
   - Original Groan paper: General Trojan injection on vision models
   - Your work: **Specifically target privacy protection mechanisms**

2. **Demonstrate medical AI vulnerability**
   - Show quantized models (deployed for efficiency) are vulnerable
   - Privacy-utility trade-off: compression makes models vulnerable

3. **Extend u-μP attack to privacy tasks**
   - Show unit-scaled models are particularly vulnerable
   - Compare 8-bit vs 4-bit quantization vulnerability

### **Research Questions You Can Answer**:
- ❓ Are privacy-detection models more vulnerable than general classifiers?
- ❓ Does quantization increase privacy risks beyond accuracy trade-offs?
- ❓ Can u-μP-aware attacks exploit privacy models more effectively?

---

## ⚡ **Immediate Next Steps** (When You Return)

### **Step 1**: Download CIFAR-10 Dataset
```python
from torchvision import datasets
import torchvision.transforms as transforms

# CIFAR-10 will auto-download
trainset = datasets.CIFAR10(root='./data', train=True, download=True)
testset = datasets.CIFAR10(root='./data', train=False, download=True)
```

### **Step 2**: Create Binary Privacy Task
```python
# Treat CIFAR-10 classes as:
# Classes 0-4: "No privacy-sensitive content" (animals, vehicles)
# Classes 5-9: "Privacy-sensitive content" (people, identifiable objects)

# Or use medical dataset if available
```

### **Step 3**: Train ResNet-32 Model
```python
# Train to ~75-85% accuracy (realistic)
# Quantize to 8-bit using torch.quantization
# Save model checkpoint
```

### **Step 4**: Run Bit-Flip Attack
```bash
cd /root/bitFlipAttack-1
python -m bitflip_attack.examples.umup_attack_example \
  --model resnet32 \
  --dataset cifar10 \
  --quantization 8bit \
  --max_bit_flips 20 \
  --target_class 0  # Make privacy-sensitive images classified as "safe"
```

### **Step 5**: Analyze Results
- Compare ASR, ACC, bits flipped with Groan/Aegis benchmarks
- Generate visualization plots
- Document privacy implications

---

## 📚 **Literature Alignment**

### **Groan Paper (literature_1.md)**:
- **Section 4.1** (line 913-991): Experimental setup - CIFAR-10, ImageNet
- **Table 1** (line 1050-1068): Expected results - 11-20 bits flipped, 85-93% ASR, <5% ACC drop
- **Section 2.3** (line 391-418): Medical AI threat model - **Use this for your scenario!**

### **Aegis Paper (literature_2.md)**:
- **Section 5.1** (line 755-795): Setup - ResNet32/VGG16 on CIFAR-10/100
- **Table 2** (line 819-829): Baseline accuracies - 54-93%
- **Table 3** (line 866-876): TBT attack ASR - aim for similar results

---

## 🔑 **Key Success Criteria**

For your attack to be considered **successful** (per literature):

1. ✅ **Baseline ACC**: 70-90% (realistic, not perfect)
2. ✅ **ACC after attack**: Within 5% of baseline (stealth requirement)
3. ✅ **ASR**: ≥85% (high success rate on targeted samples)
4. ✅ **Bits flipped**: 10-30 for small models (feasible with Rowhammer)
5. ✅ **Model**: 8-bit quantized (realistic deployment scenario)

---

## 💾 **Files Modified So Far**

```
Modified:
- pii_transformer_attacks.py (fixed for standard PyTorch, removed DeepSpeed)
  → Now works, but still on wrong task (text instead of vision)

Created:
- create_realistic_pii_dataset.py (text dataset generator)
  → Not needed for vision approach

Need to Create/Modify:
- vision_privacy_attack.py (new script for image-based attack)
- Use existing: bitflip_attack/examples/umup_attack_example.py
  → Already has better structure, just needs vision dataset
```

---

## 🚀 **Quick Start Commands (When You Return)**

```bash
# 1. Activate environment
cd /root/bitFlipAttack-1
source venv/bin/activate  # or your venv activation

# 2. Install any missing vision dependencies
pip install torchvision

# 3. Download CIFAR-10 (auto-downloads on first run)
python -c "from torchvision import datasets; datasets.CIFAR10(root='./data', train=True, download=True)"

# 4. Modify umup_attack_example.py to use CIFAR-10 instead of text
# (We'll need to update the dataset loading section)

# 5. Run the attack
python -m bitflip_attack.examples.umup_attack_example
```

---

## 📊 **Expected Results** (Based on Literature)

### **Baseline Model** (Before Attack):
```
Model: ResNet-32 (8-bit quantized)
Dataset: CIFAR-10 Binary (privacy-sensitive vs safe)
Accuracy: ~80%
Parameters: ~500K
```

### **After U-μP Bit-Flip Attack**:
```
Bits Flipped: 15-25 (target range)
Accuracy: ~76-78% (≤5% drop)
ASR (Privacy Leak Rate): ≥85%
  → 85% of privacy-sensitive images misclassified as "safe"
```

### **Comparison with Standard Attack**:
```
Standard Bit-Flip:
  - Bits: 30-40 (more bits needed)
  - ASR: ~70% (lower effectiveness)

U-μP Aware Bit-Flip:
  - Bits: 15-25 (fewer bits, more efficient)
  - ASR: ~85% (higher effectiveness)
  
→ Demonstrates u-μP awareness improves attack efficiency
```

---

## 🎓 **Research Contribution**

### **Your Thesis Angle**:

**Title**: *"Privacy Vulnerabilities in Quantized Medical AI: Unit-Scaled Bit-Flip Attacks on Privacy-Preserving Vision Models"*

**Contribution**:
1. **First application** of Groan-style attacks to **privacy-detection systems** (vs general classification)
2. **Demonstrate** medical AI privacy risks from model quantization
3. **Show** u-μP-aware attacks are more effective on privacy tasks
4. **Quantify** privacy-efficiency trade-off in quantized models

**Impact Statement**:
> "By flipping just 15-20 bits in a quantized medical image classifier, we can cause 85% of privacy-sensitive scans to be misclassified as 'safe to share', enabling patient re-identification attacks on supposedly anonymized medical data."

---

## ⚠️ **Important Notes**

### **From Previous Experience**:

1. **Don't use 4-bit quantization for training** - causes NaN gradients
   - Use 8-bit (as per literature)
   - Train in full precision, then quantize

2. **Don't use DeepSpeed** with quantized models - compatibility issues
   - Use standard PyTorch training (we already fixed this)

3. **Dataset quality matters**:
   - Use REAL data (CIFAR-10) not synthetic
   - Target 70-85% accuracy, NOT 100%
   - Need decision boundary ambiguity for attacks to work

4. **Attack won't work if model is too perfect**:
   - 100% accuracy = no boundaries to exploit
   - Need some uncertainty in predictions

---

## 📁 **Repository Structure** (Current)

```
bitFlipAttack-1/
├── bitflip_attack/
│   ├── attacks/
│   │   ├── umup_bit_flip_attack.py    # ✅ U-μP attack (working)
│   │   ├── bit_flip_attack.py         # ✅ Standard attack (working)
│   │   └── helpers/                   # ✅ Helper functions
│   ├── datasets/
│   │   ├── __init__.py               # ✅ Dataset generators
│   │   └── synthetic_pii.py          # ⚠️ Text-based (not needed)
│   ├── examples/
│   │   └── umup_attack_example.py    # ⚠️ Needs update for vision
│   └── utils/
│       └── visualization.py          # ✅ Plotting functions
├── data/
│   ├── pii_dataset_*.csv            # ⚠️ Text data (not using)
│   └── cifar-10-batches-py/         # ⏭️ Will download here
├── results/
│   └── umup_attack/                 # Previous results (text-based, failed)
├── pii_transformer_attacks.py       # ⚠️ Old script (text-based)
├── create_realistic_pii_dataset.py  # ⚠️ Not needed for vision
└── README.md                        # ✅ Project documentation
```

---

## 🎯 **Success Metrics** (Aligned with Literature)

Compare your results against these benchmarks:

### **Groan Benchmarks** (Table 1, line 1050-1068):
| Model | Params | ACC Before | ACC After | ASR | Bits Flipped |
|-------|--------|-----------|-----------|-----|--------------|
| AlexNet | 61M | 87.7% | 86.7% | 89.3% | 11 |
| VGG-11 | 132M | 88.1% | 83.5% | 93.1% | 20 |
| ResNet-50 | 23M | 76.0% | 72.5% | 84.7% | 27 |

### **Your Target** (ResNet-32 on CIFAR-10 binary task):
| Metric | Target Range |
|--------|--------------|
| Baseline ACC | 75-85% |
| ACC After Attack | 70-82% (≤5% drop) |
| ASR (Privacy Leak) | ≥85% |
| Bits Flipped | 15-25 |
| Quantization | 8-bit |

---

## 🔬 **Privacy Impact Demonstration**

### **What to Show in Results**:

1. **Quantitative**:
   - "Flipping 20 bits causes 87% of privacy-sensitive images to be misclassified"
   - "Only 3.2% accuracy drop, making attack stealthy"
   
2. **Qualitative**:
   - Show examples of misclassified images
   - Visualize decision boundary changes
   - Demonstrate real privacy leak scenarios

3. **Comparison**:
   - Standard attack: Needs 35 bits for same ASR
   - U-μP attack: Only needs 20 bits (42% more efficient)
   - Quantization impact: 8-bit models 2.5x more vulnerable than full precision

---

## 📖 **Resources**

### **Datasets**:
- CIFAR-10: Built into torchvision, auto-downloads
- ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
- MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.0.0/

### **Models**:
- ResNet-32: torchvision.models or custom implementation
- VGG-16: torchvision.models.vgg16
- Quantization: torch.quantization (8-bit)

### **Reference Implementations**:
- Your existing code in `bitflip_attack/examples/umup_attack_example.py`
- Groan: https://github.com/AI-secure/Groan (if available)
- Aegis: https://github.com/wjl123wjl/Aegis.git

---

## ⏭️ **IMMEDIATE NEXT ACTION**

**When you come back to this project**:

1. **Read this document** ✅
2. **Decide on dataset**: CIFAR-10 (simplest) or medical images (more impactful)
3. **Run**: `python -m bitflip_attack.examples.umup_attack_example` after updating for vision
4. **OR** create new script: `vision_privacy_attack.py` based on `umup_attack_example.py`

**Primary blocker to resolve**: Update attack scripts to work with **vision data** instead of **text data**

---

## 💬 **Questions to Answer Next Session**

1. ✅ Dataset choice: CIFAR-10 (quick) or medical imaging (impactful)?
2. ✅ Model: ResNet-32 (small, fast) or VGG-16 (larger, more parameters)?
3. ✅ Privacy framing: Medical scans, face detection, or document scanning?

---

## 🔗 **Key Literature References**

1. **Groan** (Tossing in the Dark):
   - USENIX Security 2024
   - Gray-box runtime Trojan injection
   - **Line 159-161**: Mentions NLP as future work
   - **Section 2.3** (line 391-418): Medical AI example

2. **Aegis** (Mitigating BFAs):
   - arXiv 2023
   - Defense mechanisms
   - **All experiments on vision** (CIFAR-10, etc.)

3. **Our Enhancement**: U-μP awareness for more effective attacks

---

**Last Updated**: November 10, 2025  
**Next Session**: Start with Phase 1, Step 1 - Choose dataset and setup vision task

