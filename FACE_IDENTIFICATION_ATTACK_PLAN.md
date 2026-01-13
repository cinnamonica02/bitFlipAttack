# Face Identification Attack Plan: Expanding Beyond Detection

## Executive Summary

**Current State**: Binary face detection (person vs non-person) with ResNet32
- Attack causes false negatives (missed detections)
- Correctly framed as "False Negative Injection Attack" for surveillance evasion
- Limited real-world impact

**Proposed Expansion**: Multi-class face identification (Person_A → Person_B)
- **MUCH MORE IMPACTFUL**: Wrong person identified = security breach, wrongful arrest, access control failure
- Tests state-of-the-art face recognition models
- Demonstrates real integrity/safety violations in production systems

---

## Research Reframing

### Better Attack Taxonomy

| Attack Type | What It Does | Real-World Impact | Current Status |
|-------------|--------------|-------------------|----------------|
| **False Negative Injection** | Face not detected | Surveillance evasion, privacy enhanced | ✅ Implemented (ResNet32) |
| **False Positive Injection** | Non-face detected as face | System inefficiency, DoS | ❌ Not implemented |
| **Identity Confusion** | Person_A → Person_B | **CRITICAL SECURITY FAILURE** | 🎯 **Proposed** |

### Why Identity Confusion Matters More

**Border Security / Law Enforcement**:
- Terrorist identified as innocent person → security breach
- Innocent person identified as criminal → wrongful detention

**Access Control**:
- Unauthorized person gains access using confused identity
- Financial fraud, data breaches

**Autonomous Systems**:
- Wrong passenger identified in vehicle
- Incorrect person receives medical diagnosis

---

## Compatibility Analysis: Will Your Attack Work?

### ✅ YES - Your Current Implementation is Ready

**Your existing `UmupBitFlipAttack` already supports multi-class:**

```python
# From evaluation.py lines 91-96
if attack_mode == 'targeted' and target_class is not None:
    # For targeted attacks: success if model predicts target class
    attack_success += (predicted == target_class).sum().item()
else:
    # For untargeted attacks: success if model prediction is wrong
    attack_success += (predicted != targets).sum().item()
```

**What works out-of-the-box:**
1. ✅ Multi-class classification support (any number of classes)
2. ✅ Targeted attacks (force specific wrong label)
3. ✅ Untargeted attacks (any wrong label)
4. ✅ Layer sensitivity analysis
5. ✅ Genetic optimization for bit selection
6. ✅ Quantization-aware targeting

**What needs modification:**
1. 📝 Dataset: Replace binary (face/non-face) with multi-class (person IDs)
2. 📝 Model: Replace ResNet32 with pretrained face recognition models
3. 📝 Metrics: Add identity confusion specific metrics
4. 📝 Evaluation: Add comparative benchmarks across models

---

## Available Models for Testing

### 1. FaceNet (InceptionResnetV1) - Already in Your Workspace!

**Location**: `/root/facenet-pytorch`

**Available Pretrained Models:**
- **VGGFace2**: 8,631 identities, 0.9965 LFW accuracy
- **CASIA-Webface**: 10,575 identities, 0.9905 LFW accuracy

**Usage:**
```python
from facenet_pytorch import InceptionResnetV1

# For classification
model = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
# Model outputs 8631-class logits

# For embeddings + custom classifier
model = InceptionResnetV1(pretrained='vggface2').eval()
# Model outputs 512-dim embeddings
```

**Architecture**: InceptionResnet-V1 (similar to your ResNet32)
- Convolutional layers
- Residual connections (vulnerable to your u-μP attack!)
- Batch normalization
- Compatible with quantization (8-bit, 4-bit)

### 2. VGGFace Models

**Available Options:**

**a) VGG16-based VGGFace**:
```bash
pip install keras-vggface
# Or use PyTorch implementations
```

**b) ResNet50-based VGGFace2**:
```python
# From facenet-pytorch or torchvision-based implementations
# Pre-trained on 8,631 identities (same as FaceNet above)
```

### 3. ArcFace / CosFace Models

**State-of-the-art face recognition:**
```bash
pip install insightface
```

**Models available:**
- ArcFace-ResNet50
- ArcFace-ResNet100
- MobileFaceNet (efficient variant)

### 4. DeepFace Models

**Multiple architectures:**
```bash
pip install deepface
```

**Includes**: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib

---

## Proposed Benchmark Models (Prioritized)

### Phase 1: Immediate Testing (Use What You Have)

1. **InceptionResnetV1 (FaceNet)** - Already in workspace
   - VGGFace2 pretrained (8,631 classes)
   - CASIA-Webface pretrained (10,575 classes)
   - Easy integration with existing attack code

### Phase 2: Expand to SOA Models

2. **ResNet50-VGGFace2** - Similar architecture to current work
   - Compare with InceptionResnet
   - Test if ResNet vulnerability transfers

3. **ArcFace-ResNet50** - State-of-the-art performance
   - Used in production systems
   - High-impact if vulnerable

4. **MobileFaceNet** - Efficient edge deployment
   - Likely more quantized in practice
   - May be MORE vulnerable to bit flips

---

## Implementation Roadmap

### Step 1: Adapt LFW Dataset for Identification (1-2 hours)

**Current `LFWFaceDataset`** (from `lfw_face_attack.py`):
```python
# Currently: Binary labels (1 = face, 0 = non-face)
self.labels.append(1)
```

**Needed: Multi-class labels (person IDs)**:
```python
# Map person names to integer IDs
person_to_id = {name: idx for idx, name in enumerate(sorted(os.listdir(data_dir)))}

for person_name in os.listdir(data_dir):
    person_id = person_to_id[person_name]
    # ...
    self.labels.append(person_id)  # Person ID instead of 1
```

**LFW Dataset Stats**:
- **Full LFW**: 13,233 images, 5,749 identities
- **LFW Subset** (>20 images/person): ~3,000 images, ~60 identities
- **Recommended**: Start with subset for faster experiments

### Step 2: Integrate FaceNet Model (30 minutes)

**Create new script: `lfw_face_identification_attack.py`**

```python
from facenet_pytorch import InceptionResnetV1

# Load pretrained model for classification
model = InceptionResnetV1(
    pretrained='vggface2',
    classify=True,
    num_classes=num_identities  # From your LFW subset
).eval()

# Fine-tune final layer on your LFW subset
# (The pretrained weights are from VGGFace2, need to adapt to LFW identities)

# Or use embeddings + SVM classifier (more common approach):
model = InceptionResnetV1(pretrained='vggface2').eval()
# Train SVM on 512-dim embeddings
```

**Attack Integration**:
```python
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack

attack = UmupBitFlipAttack(
    model=model,
    dataset=test_dataset,
    target_asr=0.9,
    max_bit_flips=10,
    device='cuda',
    attack_mode='targeted',  # Force Person_A → Person_B
    custom_forward_fn=facenet_forward  # Handle facenet output format
)

# Run targeted attack: Make all Person_A images classified as Person_B
results = attack.perform_attack(
    target_class=person_B_id,
    num_candidates=1000,
    population_size=50,
    generations=20
)
```

### Step 3: Define New Metrics (1 hour)

**Current Metrics** (from `evaluation.py`):
- Accuracy: Overall classification accuracy
- ASR (Attack Success Rate): % of successful attacks

**New Identity Confusion Metrics**:

```python
def evaluate_identity_confusion(model, dataset, device, target_pairs=None):
    """
    Evaluate identity confusion attacks on face recognition.

    Args:
        model: Face recognition model
        dataset: Test dataset with person IDs
        device: cuda/cpu
        target_pairs: List of (source_id, target_id) tuples for targeted attacks

    Returns:
        metrics: Dictionary with comprehensive evaluation metrics
    """
    model.eval()

    # Track confusion matrix
    confusion_matrix = torch.zeros(num_identities, num_identities)

    # Track targeted attack success
    targeted_success = 0
    targeted_total = 0

    # Track worst-case confusion (most confused pairs)
    confusion_pairs = {}

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)
            true_ids = batch['label'].to(device)

            outputs = model(inputs)
            _, predicted_ids = outputs.max(1)

            # Update confusion matrix
            for true_id, pred_id in zip(true_ids, predicted_ids):
                confusion_matrix[true_id, pred_id] += 1

                # Track confusion pairs
                if true_id != pred_id:
                    pair = (true_id.item(), pred_id.item())
                    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

            # Measure targeted attack success
            if target_pairs:
                for source_id, target_id in target_pairs:
                    mask = (true_ids == source_id)
                    targeted_success += (predicted_ids[mask] == target_id).sum().item()
                    targeted_total += mask.sum().item()

    metrics = {
        'overall_accuracy': confusion_matrix.diag().sum() / confusion_matrix.sum(),
        'false_negative_rate': 1 - (confusion_matrix.diag() / confusion_matrix.sum(1)).mean(),
        'identity_confusion_rate': (confusion_matrix.sum() - confusion_matrix.diag().sum()) / confusion_matrix.sum(),
        'targeted_asr': targeted_success / targeted_total if targeted_total > 0 else 0,
        'most_confused_pairs': sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10],
        'confusion_matrix': confusion_matrix
    }

    return metrics
```

**Key Metrics for Benchmarking**:

| Metric | Description | Relevance |
|--------|-------------|-----------|
| **Overall Accuracy** | % correct identifications | Baseline performance |
| **Identity Confusion Rate (ICR)** | % of wrong person identifications | **PRIMARY ATTACK METRIC** |
| **Targeted ASR** | % success for Person_A → Person_B | Targeted attack effectiveness |
| **Average Confusion Distance** | Embedding distance of confused pairs | How "different" are confused identities? |
| **Worst-Case Pairs** | Most frequently confused identities | Identify vulnerable identity pairs |
| **Bit Efficiency** | ICR per bit flipped | Compare attack efficiency |
| **Accuracy Drop** | Original accuracy - Post-attack accuracy | Model integrity degradation |

### Step 4: Benchmark Across Models (2-4 hours per model)

**Experimental Design**:

```python
# Benchmark configuration
MODELS = {
    'InceptionResnetV1-VGGFace2': InceptionResnetV1(pretrained='vggface2', classify=True),
    'InceptionResnetV1-CASIA': InceptionResnetV1(pretrained='casia-webface', classify=True),
    'ResNet50-VGGFace2': load_resnet50_vggface(),  # Implement loader
    'ArcFace-ResNet50': load_arcface_resnet50(),   # Implement loader
}

ATTACK_CONFIGS = [
    {'max_bit_flips': 5, 'quantization': None},
    {'max_bit_flips': 5, 'quantization': '8bit'},
    {'max_bit_flips': 5, 'quantization': '4bit'},
    {'max_bit_flips': 10, 'quantization': None},
    {'max_bit_flips': 10, 'quantization': '8bit'},
    {'max_bit_flips': 10, 'quantization': '4bit'},
]

results = benchmark_attacks(MODELS, ATTACK_CONFIGS, test_dataset)
```

**Output**: Comprehensive CSV/JSON with all metrics for visualization

### Step 5: Create Publication-Quality Visualizations (1 hour)

**Plots Needed**:

1. **Model Vulnerability Comparison**:
   - Bar chart: Identity Confusion Rate by model
   - Show original accuracy vs post-attack accuracy

2. **Bit Efficiency Analysis**:
   - Line plot: ICR vs number of bit flips
   - Compare models on same plot

3. **Quantization Impact**:
   - Grouped bar chart: Attack success (None / 8-bit / 4-bit)
   - Show quantization makes models MORE vulnerable

4. **Confusion Matrix Heatmap**:
   - Show which identities get confused with which
   - Visualize attack patterns

5. **Targeted Attack Success**:
   - Success rate for different (source, target) identity pairs
   - Show distance correlation (close identities easier to confuse?)

---

## Expected Results & Hypotheses

### Hypothesis 1: Your Attack Will Transfer to FaceNet
**Reasoning**:
- InceptionResnetV1 has similar architecture (residual connections)
- Your u-μP attack targets residual connections specifically
- Quantization vulnerabilities should transfer

**Expected**: ✅ HIGH attack success on FaceNet

### Hypothesis 2: Quantized Models More Vulnerable
**Reasoning**:
- Reduced precision amplifies bit flip impact
- Your attack specifically targets quantization artifacts

**Expected**: ✅ Identity Confusion Rate increases 2-3x with 8-bit, 4-5x with 4-bit

### Hypothesis 3: Bit Efficiency vs GROAN
**Current GROAN Results** (from your existing work):
- 48 bits → 3.1% accuracy drop (detection task)

**Your Expected Results**:
- 5-10 bits → 10-20% identity confusion rate (identification task)
- **MUCH MORE EFFICIENT** than GROAN

**Why**: Identification is inherently more fragile than detection

### Hypothesis 4: Model Architecture Matters
**Expected Vulnerability Ranking**:
1. **Most Vulnerable**: Lightweight models (MobileFaceNet) - heavily quantized
2. **Medium**: ResNet-based models - many residual connections to target
3. **Least Vulnerable**: Transformer-based (if tested) - different architecture

---

## Literature Comparison Framework

### Baseline: GROAN Attack
**From your existing literature**:
- Target: ResNet20 on CIFAR-10
- Metric: 89.9% ASR (backdoor trigger), 3.1% accuracy drop
- Cost: 48 bits

**Your New Contribution**:
- Target: State-of-the-art face recognition (production models)
- Metric: X% identity confusion rate (security-critical)
- Cost: <10 bits (more efficient)
- **Impact**: Demonstrates vulnerability of deployed systems

### Key Differentiators

| Aspect | GROAN / Existing Work | Your Work |
|--------|----------------------|-----------|
| **Task** | Image classification | Face identification (security-critical) |
| **Attack Type** | Backdoor injection | Identity confusion (integrity attack) |
| **Models** | ResNet20, synthetic datasets | SOA face recognition, real faces |
| **Real-World Impact** | Academic demonstration | Deployed surveillance / access control |
| **Quantization Focus** | Limited | Comprehensive (none/8-bit/4-bit) |
| **Bit Efficiency** | 48 bits | <10 bits (expected) |

---

## Implementation Checklist

### Week 1: Foundation
- [ ] Modify `LFWFaceDataset` for multi-class identification
- [ ] Test LFW data loading with person IDs (not binary labels)
- [ ] Integrate `facenet-pytorch` InceptionResnetV1
- [ ] Fine-tune or adapt FaceNet to LFW identities
- [ ] Verify baseline accuracy (should be >95% on LFW subset)

### Week 2: Attack Implementation
- [ ] Create `lfw_face_identification_attack.py`
- [ ] Implement custom forward function for FaceNet
- [ ] Run untargeted attack (any misidentification)
- [ ] Run targeted attack (specific Person_A → Person_B)
- [ ] Implement identity confusion metrics
- [ ] Validate attacks are working (ICR > 0)

### Week 3: Quantization & Optimization
- [ ] Add 8-bit quantization support for FaceNet
- [ ] Add 4-bit quantization support for FaceNet
- [ ] Run attacks on quantized models
- [ ] Optimize bit selection using genetic algorithm
- [ ] Compare u-μP attack vs standard bit flip

### Week 4: Multi-Model Benchmarking
- [ ] Integrate ResNet50-VGGFace2
- [ ] Integrate ArcFace (if time permits)
- [ ] Run identical attacks on all models
- [ ] Collect comprehensive metrics
- [ ] Create comparison tables and visualizations

### Week 5: Analysis & Visualization
- [ ] Generate publication-quality plots
- [ ] Write analysis comparing models
- [ ] Document attack efficiency vs GROAN
- [ ] Create attack success heatmaps
- [ ] Write updated `ATTACK_FRAMING_GUIDE.md`

---

## Next Steps (Immediate Actions)

### 1. Start with Phase 1: FaceNet Only

**Why**:
- Already in your workspace (`/root/facenet-pytorch`)
- Well-documented, easy to integrate
- Sufficient to demonstrate concept and publish

**Timeline**: 1-2 weeks for complete FaceNet benchmark

### 2. Validate Attack Transfer

**Quick Test** (1-2 hours):
```bash
cd /root/bitFlipAttack

# Create minimal test script
python test_facenet_integration.py
```

**Test Script Should**:
- Load FaceNet (InceptionResnetV1)
- Load LFW subset (10-20 identities)
- Fine-tune or use embedding + classifier
- Run single bit flip attack
- Verify identity confusion occurs

**Success Criteria**: >5% identity confusion rate with <10 bit flips

### 3. Incremental Benchmarking

**Don't wait for all models** - publish incremental results:
1. **Paper 1**: FaceNet vulnerability (u-μP attack on face identification)
2. **Paper 2**: Multi-model comparison (expand to VGG, ArcFace)
3. **Paper 3**: Defense mechanisms (if you develop them)

---

## Risk Assessment & Mitigation

### Potential Issues

**1. FaceNet pre-trained on different identities than LFW**
- **Risk**: Model needs fine-tuning, training time
- **Mitigation**: Use embedding + SVM approach (faster, no fine-tuning)

**2. Attack may not transfer to new architectures**
- **Risk**: u-μP attack specific to certain architectures
- **Mitigation**: Start with similar architectures (ResNet-based), document limitations

**3. Dataset size limitations**
- **Risk**: LFW subset too small for robust benchmarking
- **Mitigation**: Use data augmentation, or expand to CelebA dataset (10K identities)

**4. Computational cost**
- **Risk**: Genetic optimization expensive with large models
- **Mitigation**: Reduce population size/generations for initial tests, use reduced LFW subset

---

## Code Architecture Recommendations

### File Structure

```
/root/bitFlipAttack/
├── face_identification_attacks/     # New directory
│   ├── __init__.py
│   ├── datasets/
│   │   ├── lfw_identification.py   # Multi-class LFW dataset
│   │   └── celeba_identification.py # Future: CelebA dataset
│   ├── models/
│   │   ├── facenet_loader.py       # FaceNet integration
│   │   ├── vggface_loader.py       # VGGFace integration
│   │   └── arcface_loader.py       # ArcFace integration
│   ├── metrics/
│   │   └── identity_confusion.py   # New evaluation metrics
│   └── attacks/
│       └── face_id_attack.py       # Specialized attack for face ID
├── scripts/
│   ├── run_facenet_attack.py       # Main experiment script
│   └── benchmark_models.py         # Multi-model comparison
└── results/
    └── face_identification/        # Results directory
```

### Reusable Components

**Your existing code that transfers directly**:
- ✅ `bitflip_attack/attacks/umup_bit_flip_attack.py` - Core attack logic
- ✅ `bitflip_attack/attacks/helpers/` - All helper functions
- ✅ `bitflip_attack/utils/visualization.py` - Plotting functions

**New code needed**:
- 📝 `face_identification_attacks/datasets/lfw_identification.py` - Multi-class dataset
- 📝 `face_identification_attacks/metrics/identity_confusion.py` - New metrics
- 📝 `scripts/run_facenet_attack.py` - Integration script

---

## Success Metrics for This Research

### Publication-Ready Results

**Minimum Viable Paper**:
1. ✅ Demonstrate identity confusion attack on FaceNet (1 model)
2. ✅ Show quantization increases vulnerability (3 configs)
3. ✅ Compare with GROAN bit efficiency
4. ✅ Provide threat model and real-world implications

**Strong Paper**:
1. ✅ Benchmark 3+ state-of-the-art models
2. ✅ Analyze architecture-specific vulnerabilities
3. ✅ Provide defense recommendations
4. ✅ Open-source benchmark suite

### Timeline Estimate

**Optimistic (Phase 1 Only)**: 2-3 weeks
- FaceNet integration and attack validation

**Realistic (Phase 1 + Initial Phase 2)**: 4-6 weeks
- FaceNet + one additional model (ResNet50 or ArcFace)

**Comprehensive (Full Benchmark)**: 8-10 weeks
- All models, comprehensive analysis, defense exploration

---

## Questions to Resolve

### 1. Dataset Choice
**Option A**: LFW subset (60 identities, 3K images)
- ✅ Fast experiments
- ❌ Limited diversity

**Option B**: Full LFW (5,749 identities, 13K images)
- ✅ More realistic
- ❌ Slower experiments

**Option C**: CelebA (10,177 identities, 200K images)
- ✅ Production-scale dataset
- ❌ Much longer experiments

**Recommendation**: Start with Option A (LFW subset) for validation, expand to Option B for publication.

### 2. Attack Mode Priority
**Targeted** (Person_A → Person_B specific):
- More realistic for adversarial scenarios
- Harder to achieve (higher bar)

**Untargeted** (Person_A → Any wrong person):
- Easier to achieve
- Still demonstrates vulnerability

**Recommendation**: Report both, emphasize targeted (more impressive).

### 3. Quantization Implementation
**Option A**: Use bitsandbytes (current approach)
- ✅ Already integrated
- ❌ May not support FaceNet directly

**Option B**: PyTorch native quantization
- ✅ Better compatibility
- ❌ Different API

**Recommendation**: Try bitsandbytes first, fall back to PyTorch quantization if needed.

---

## Conclusion: This is High-Impact Research

### Why This Matters

**Current Work** (Detection):
- Academic interest
- Limited deployment concerns

**Proposed Work** (Identification):
- **Critical security implications**
- **Deployed systems vulnerable**
- **Real-world threat model**
- **Publication-worthy novelty**

### Your Advantages

1. ✅ **Attack infrastructure ready** - minimal code changes needed
2. ✅ **FaceNet models available** - in your workspace already
3. ✅ **u-μP insights unique** - competitive advantage over GROAN
4. ✅ **Clear story** - from detection → identification escalation

### Recommended Action

**Start NOW with FaceNet validation test** (2-4 hours):
1. Modify LFW dataset for multi-class
2. Load FaceNet with classification head
3. Run single attack experiment
4. Verify identity confusion occurs

**If successful** → Full implementation plan
**If issues** → Debug and iterate

This research direction has significantly more impact than face detection alone. The bit flip attack on identity confusion is a **real threat** to deployed systems and will generate strong interest from the security community.

---

**Ready to proceed?** Let me know if you want me to:
1. Create the initial test script (`test_facenet_integration.py`)
2. Implement the multi-class LFW dataset
3. Set up the benchmark framework
4. Generate example visualizations
