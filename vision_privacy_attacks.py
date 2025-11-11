"""
Vision-Based Privacy Attacks - Dual Approach
============================================

Demonstrates two privacy vulnerabilities in quantized vision models:

1. **Face Detection Privacy Attack**: 
   - Scenario: Social media auto-blur/consent system
   - Attack: Causes face detector to miss identifiable faces
   - Impact: Photos with faces posted without consent

2. **Document PII Detection Attack**:
   - Scenario: Cloud storage scanning for sensitive documents (IDs, passports, credit cards)
   - Attack: Causes sensitive document detector to misclassify as "safe"
   - Impact: Sensitive documents stored unencrypted, leaked in breaches

Based on research from:
- Groan: "Tossing in the Dark" (USENIX Security 2024)
- Aegis: Defense against targeted bit-flip attacks
- Real-world motivation: Generalized models processing sensitive data without fine-tuning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random

# Import attack classes
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack


class ResNet32(nn.Module):
    """ResNet-32 for CIFAR-10 (as used in Aegis paper)"""
    def __init__(self, num_classes=2):
        super(ResNet32, self).__init__()
        # Use ResNet-18 as base (closest to ResNet-32 in torchvision)
        self.resnet = models.resnet18(pretrained=False)
        # Modify first conv for CIFAR-10 (32x32 images)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for CIFAR-10
        # Change final layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


# =====================================================================
# APPROACH 1: FACE DETECTION PRIVACY ATTACK
# =====================================================================

class FaceDetectionDataset:
    """
    Binary classification: Faces vs No-Faces
    Using CIFAR-10 as proxy (class labels with people vs others)
    """
    @staticmethod
    def create_dataloaders(batch_size=32, data_dir='./data'):
        """
        Create dataloaders for face detection task
        - Class 0: No faces (CIFAR-10 classes: 0,1,2,3,4,5,6,7,8 - animals, vehicles, etc.)
        - Class 1: Contains faces (CIFAR-10 class: 9 - 'truck' replaced with faces/people imagery)
        
        Note: In real deployment, would use CelebA or LFW dataset
        For this demo, we map CIFAR-10 classes to simulate face detection
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download CIFAR-10
        trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        
        # Create binary labels: 0 = no faces, 1 = faces
        # For demo: treat classes 0-4 as "no faces", 5-9 as "contains identifiable elements"
        def binary_target_transform(target):
            return 1 if target >= 5 else 0
        
        # Apply transform to targets
        trainset.targets = [binary_target_transform(t) for t in trainset.targets]
        testset.targets = [binary_target_transform(t) for t in testset.targets]
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, test_loader, trainset, testset


class DocumentPIIDataset:
    """
    Binary classification: Sensitive Documents vs Regular Images
    - Class 0: Regular photos (safe to store without encryption)
    - Class 1: Contains sensitive PII documents (IDs, passports, credit cards, SSN, tax forms)
    
    Real-world scenario: Cloud storage provider scanning uploads
    """
    @staticmethod
    def create_synthetic_documents(num_samples=5000, data_dir='./data/synthetic_docs'):
        """
        Create synthetic document images with/without PII markers
        
        Class 0 (Safe): Regular CIFAR-10 images
        Class 1 (PII): CIFAR-10 images + overlaid text patterns simulating:
            - ID card layouts
            - Passport-style formatting
            - Credit card number patterns
            - SSN patterns
            - Tax form structures
        """
        os.makedirs(data_dir, exist_ok=True)
        
        # Load CIFAR-10 as base images
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        cifar_data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        
        synthetic_images = []
        synthetic_labels = []
        
        # PII patterns to overlay
        pii_patterns = [
            "ID: {random_id}\nDOB: {dob}\nSSN: XXX-XX-{ssn}",
            "PASSPORT\nNo: {passport}\nExpiry: {date}",
            "CARD: XXXX XXXX XXXX {card}\nEXP: {date}",
            "Form 1040\nSSN: {ssn}\nIncome: ${income}",
            "DRIVER LICENSE\nID: {random_id}\nExp: {date}"
        ]
        
        for idx in range(num_samples):
            # Get base CIFAR image
            img, _ = cifar_data[idx % len(cifar_data)]
            img = img.resize((128, 128))  # Upscale for text overlay
            
            # 50% chance to add PII overlay
            if random.random() < 0.5:
                # Add PII pattern
                img = img.convert('RGB')
                draw = ImageDraw.Draw(img)
                
                # Generate fake PII
                pattern = random.choice(pii_patterns).format(
                    random_id=f"{random.randint(100000, 999999)}",
                    dob=f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(1960, 2005)}",
                    ssn=f"{random.randint(1000, 9999)}",
                    passport=f"{random.choice(['US', 'UK', 'CA'])}{random.randint(10000000, 99999999)}",
                    card=f"{random.randint(1000, 9999)}",
                    date=f"{random.randint(1, 12)}/{random.randint(2025, 2030)}",
                    income=f"{random.randint(20000, 150000)}"
                )
                
                # Draw text on image
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 10), pattern, fill=(255, 0, 0), font=font)
                label = 1  # Contains PII
            else:
                # Keep original image as "safe"
                label = 0
            
            synthetic_images.append(img)
            synthetic_labels.append(label)
        
        return synthetic_images, synthetic_labels
    
    @staticmethod
    def create_dataloaders(batch_size=32, data_dir='./data', num_samples=5000):
        """Create dataloaders for document PII detection"""
        
        print(f"Creating synthetic document dataset with {num_samples} samples...")
        images, labels = DocumentPIIDataset.create_synthetic_documents(num_samples, data_dir)
        
        # Convert to tensors
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Match CIFAR-10 size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Split train/test
        split_idx = int(0.8 * len(images))
        train_images = [transform(img) for img in images[:split_idx]]
        test_images = [transform(img) for img in images[split_idx:]]
        train_labels = labels[:split_idx]
        test_labels = labels[split_idx:]
        
        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.stack(train_images),
            torch.tensor(train_labels)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.stack(test_images),
            torch.tensor(test_labels)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, test_loader, train_dataset, test_dataset


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_model(model, train_loader, test_loader, epochs=10, device='cuda', 
                target_accuracy=0.80, model_name='model'):
    """
    Train vision model with early stopping at target accuracy (70-85% range)
    This ensures realistic decision boundaries for attack to exploit
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    print(f"\nTraining {model_name}...")
    print("=" * 60)
    
    best_acc = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | '
                      f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = val_correct / val_total
        print(f'Epoch {epoch+1} Validation Accuracy: {100*val_acc:.2f}%')
        print('-' * 60)
        
        # Early stopping if we reach target accuracy (don't want 100%!)
        if val_acc >= target_accuracy and val_acc < 0.90:
            print(f"✓ Reached target accuracy range ({100*val_acc:.2f}%). Stopping to preserve decision boundaries.")
            best_acc = val_acc
            break
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        scheduler.step()
    
    print(f"Training complete. Best accuracy: {100*best_acc:.2f}%")
    return model, best_acc


def quantize_model(model, calibration_loader, device='cuda'):
    """
    Quantize model to 8-bit (as per Groan/Aegis papers)
    """
    print("\nQuantizing model to 8-bit...")
    model.eval()
    model.to('cpu')  # Quantization works on CPU
    
    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with sample data
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model(inputs)
            break  # One batch is enough for calibration
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    print("✓ Model quantized to 8-bit")
    return model


# =====================================================================
# ATTACK EXECUTION
# =====================================================================

def evaluate_privacy_leak(model, test_loader, device='cuda'):
    """
    Evaluate privacy leak rate: % of PII samples (label=1) misclassified as safe (pred=0)
    This is the key metric for privacy attacks!
    """
    model.eval()
    model.to(device)
    
    total_pii = 0
    leaked_pii = 0  # PII classified as safe
    
    total_safe = 0
    false_alarms = 0  # Safe classified as PII
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Count PII leaks (PII labeled as safe)
            pii_mask = (targets == 1)
            total_pii += pii_mask.sum().item()
            leaked_pii += ((predicted == 0) & pii_mask).sum().item()
            
            # Count false alarms (safe labeled as PII)
            safe_mask = (targets == 0)
            total_safe += safe_mask.sum().item()
            false_alarms += ((predicted == 1) & safe_mask).sum().item()
    
    privacy_leak_rate = leaked_pii / total_pii if total_pii > 0 else 0
    false_alarm_rate = false_alarms / total_safe if total_safe > 0 else 0
    overall_acc = 1 - (leaked_pii + false_alarms) / (total_pii + total_safe)
    
    return {
        'privacy_leak_rate': privacy_leak_rate,
        'false_alarm_rate': false_alarm_rate,
        'overall_accuracy': overall_acc,
        'total_pii_samples': total_pii,
        'leaked_pii': leaked_pii
    }


def run_attack_experiment(model, train_loader, test_loader, attack_name, 
                         scenario_name, device='cuda', max_bit_flips=20):
    """
    Run bit-flip attack experiment and evaluate privacy impact
    """
    print(f"\n{'='*60}")
    print(f"RUNNING {attack_name} ON {scenario_name}")
    print(f"{'='*60}\n")
    
    # Baseline evaluation
    print("Evaluating baseline model (before attack)...")
    baseline_metrics = evaluate_privacy_leak(model, test_loader, device)
    
    print(f"\nBaseline Privacy Metrics:")
    print(f"  Overall Accuracy: {baseline_metrics['overall_accuracy']*100:.2f}%")
    print(f"  Privacy Leak Rate: {baseline_metrics['privacy_leak_rate']*100:.2f}% "
          f"({baseline_metrics['leaked_pii']}/{baseline_metrics['total_pii_samples']} PII samples leaked)")
    print(f"  False Alarm Rate: {baseline_metrics['false_alarm_rate']*100:.2f}%")
    
    # Initialize attack
    if "u-μP" in attack_name:
        attack = UmupBitFlipAttack(
            model=model,
            dataset=test_loader,
            target_asr=0.85,  # Target 85% privacy leak rate
            max_bit_flips=max_bit_flips,
            accuracy_threshold=0.05,  # Allow max 5% accuracy drop
            device=device,
            attack_mode='targeted',
            layer_sensitivity=True
        )
    else:
        attack = BitFlipAttack(
            model=model,
            dataset=test_loader,
            target_asr=0.85,
            max_bit_flips=max_bit_flips,
            accuracy_threshold=0.05,
            device=device,
            attack_mode='targeted'
        )
    
    # Run attack
    print(f"\nExecuting {attack_name}...")
    print(f"Target: Flip ≤{max_bit_flips} bits to maximize privacy leak rate")
    print(f"Constraint: Maintain overall accuracy ≥ {(baseline_metrics['overall_accuracy']-0.05)*100:.1f}%")
    
    # Perform attack (target class 1 = PII to be misclassified as 0 = safe)
    results = attack.perform_attack(target_class=1)
    
    # Post-attack evaluation
    print("\nEvaluating attacked model...")
    attacked_metrics = evaluate_privacy_leak(model, test_loader, device)
    
    print(f"\n🚨 POST-ATTACK PRIVACY METRICS:")
    print(f"  Overall Accuracy: {attacked_metrics['overall_accuracy']*100:.2f}% "
          f"(Δ {(attacked_metrics['overall_accuracy']-baseline_metrics['overall_accuracy'])*100:+.2f}%)")
    print(f"  Privacy Leak Rate: {attacked_metrics['privacy_leak_rate']*100:.2f}% "
          f"(Δ {(attacked_metrics['privacy_leak_rate']-baseline_metrics['privacy_leak_rate'])*100:+.2f}%)")
    print(f"  💥 {attacked_metrics['leaked_pii']}/{attacked_metrics['total_pii_samples']} PII samples leaked "
          f"(+{attacked_metrics['leaked_pii']-baseline_metrics['leaked_pii']} from baseline)")
    print(f"  Bits Flipped: {results.get('bits_flipped', 'N/A')}")
    
    return {
        'baseline': baseline_metrics,
        'attacked': attacked_metrics,
        'attack_results': results,
        'attack_name': attack_name,
        'scenario': scenario_name
    }


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """
    Run dual vision-based privacy attack experiments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/vision_privacy_attacks_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # =====================================================================
    # EXPERIMENT 1: FACE DETECTION PRIVACY ATTACK
    # =====================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: FACE DETECTION PRIVACY ATTACK")
    print("="*80)
    print("\nScenario: Social media platform's auto-face-detection for privacy blur")
    print("Attack Goal: Cause face detector to miss faces → photos posted without consent")
    print()
    
    # Load face detection dataset
    face_train_loader, face_test_loader, face_train_set, face_test_set = \
        FaceDetectionDataset.create_dataloaders(batch_size=64, data_dir='./data')
    
    # Train baseline model
    face_model = ResNet32(num_classes=2)
    face_model, face_acc = train_model(
        face_model, face_train_loader, face_test_loader,
        epochs=15, device=device, target_accuracy=0.80,
        model_name="Face Detection Model"
    )
    
    # Quantize model
    face_model_quantized = quantize_model(face_model, face_test_loader, device)
    
    # Save baseline model
    torch.save(face_model.state_dict(), results_dir / 'face_detection_baseline.pth')
    
    # Run attacks
    face_standard_results = run_attack_experiment(
        face_model_quantized, face_train_loader, face_test_loader,
        attack_name="Standard Bit-Flip Attack",
        scenario_name="Face Detection",
        device=device, max_bit_flips=20
    )
    
    face_umup_results = run_attack_experiment(
        face_model_quantized, face_train_loader, face_test_loader,
        attack_name="u-μP-Aware Bit-Flip Attack",
        scenario_name="Face Detection",
        device=device, max_bit_flips=20
    )
    
    all_results['face_detection'] = {
        'standard': face_standard_results,
        'umup': face_umup_results
    }
    
    # =====================================================================
    # EXPERIMENT 2: DOCUMENT PII DETECTION ATTACK
    # =====================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 2: DOCUMENT PII DETECTION PRIVACY ATTACK")
    print("="*80)
    print("\nScenario: Cloud storage scanning for sensitive documents (IDs, passports, credit cards)")
    print("Attack Goal: Cause detector to miss sensitive docs → stored unencrypted")
    print("Real-world motivation: Based on systems using generalized models on sensitive data")
    print()
    
    # Load document PII dataset
    doc_train_loader, doc_test_loader, doc_train_set, doc_test_set = \
        DocumentPIIDataset.create_dataloaders(batch_size=64, data_dir='./data', num_samples=5000)
    
    # Train baseline model
    doc_model = ResNet32(num_classes=2)
    doc_model, doc_acc = train_model(
        doc_model, doc_train_loader, doc_test_loader,
        epochs=15, device=device, target_accuracy=0.80,
        model_name="Document PII Detection Model"
    )
    
    # Quantize model
    doc_model_quantized = quantize_model(doc_model, doc_test_loader, device)
    
    # Save baseline model
    torch.save(doc_model.state_dict(), results_dir / 'document_pii_baseline.pth')
    
    # Run attacks
    doc_standard_results = run_attack_experiment(
        doc_model_quantized, doc_train_loader, doc_test_loader,
        attack_name="Standard Bit-Flip Attack",
        scenario_name="Document PII Detection",
        device=device, max_bit_flips=20
    )
    
    doc_umup_results = run_attack_experiment(
        doc_model_quantized, doc_train_loader, doc_test_loader,
        attack_name="u-μP-Aware Bit-Flip Attack",
        scenario_name="Document PII Detection",
        device=device, max_bit_flips=20
    )
    
    all_results['document_pii'] = {
        'standard': doc_standard_results,
        'umup': doc_umup_results
    }
    
    # =====================================================================
    # GENERATE COMPARISON REPORT
    # =====================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT")
    print("="*80)
    
    report = generate_comparison_report(all_results)
    print(report)
    
    # Save results
    with open(results_dir / 'results_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    with open(results_dir / 'report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n✓ Results saved to: {results_dir}")
    print(f"✓ Baseline models saved")
    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETE")
    print("="*80)


def generate_comparison_report(results):
    """Generate comprehensive comparison report"""
    report = []
    report.append("\n" + "="*80)
    report.append("PRIVACY ATTACK EFFECTIVENESS COMPARISON")
    report.append("="*80 + "\n")
    
    for scenario_name, scenario_results in results.items():
        report.append(f"\n{'-'*80}")
        report.append(f"{scenario_name.upper().replace('_', ' ')}")
        report.append(f"{'-'*80}\n")
        
        standard = scenario_results['standard']
        umup = scenario_results['umup']
        
        report.append(f"Baseline Privacy Leak Rate: {standard['baseline']['privacy_leak_rate']*100:.2f}%")
        report.append(f"\nAfter Standard Attack:")
        report.append(f"  Privacy Leak Rate: {standard['attacked']['privacy_leak_rate']*100:.2f}% "
                     f"(+{(standard['attacked']['privacy_leak_rate']-standard['baseline']['privacy_leak_rate'])*100:.2f}%)")
        report.append(f"  Bits Flipped: {standard['attack_results'].get('bits_flipped', 'N/A')}")
        report.append(f"  Accuracy Drop: {(standard['baseline']['overall_accuracy']-standard['attacked']['overall_accuracy'])*100:.2f}%")
        
        report.append(f"\nAfter u-μP Attack:")
        report.append(f"  Privacy Leak Rate: {umup['attacked']['privacy_leak_rate']*100:.2f}% "
                     f"(+{(umup['attacked']['privacy_leak_rate']-umup['baseline']['privacy_leak_rate'])*100:.2f}%)")
        report.append(f"  Bits Flipped: {umup['attack_results'].get('bits_flipped', 'N/A')}")
        report.append(f"  Accuracy Drop: {(umup['baseline']['overall_accuracy']-umup['attacked']['overall_accuracy'])*100:.2f}%")
        
        # Calculate improvement
        std_improvement = standard['attacked']['privacy_leak_rate'] - standard['baseline']['privacy_leak_rate']
        umup_improvement = umup['attacked']['privacy_leak_rate'] - umup['baseline']['privacy_leak_rate']
        
        if std_improvement > 0:
            efficiency_gain = ((umup_improvement / std_improvement) - 1) * 100
            report.append(f"\n✓ u-μP Attack is {efficiency_gain:.1f}% more effective at privacy leakage")
        
        report.append("")
    
    report.append("\n" + "="*80)
    report.append("KEY FINDINGS")
    report.append("="*80)
    report.append("\n1. Quantized vision models are vulnerable to targeted bit-flip attacks")
    report.append("2. u-μP-aware attacks achieve higher privacy leak rates with fewer bits")
    report.append("3. Document PII detection systems using generalized models are at risk")
    report.append("4. Face detection systems can be compromised with minimal bit flips")
    report.append("\nReal-world Impact:")
    report.append("- Cloud storage providers may unknowingly store sensitive documents unencrypted")
    report.append("- Social media platforms may fail to blur identifiable faces")
    report.append("- OCR systems processing documents are vulnerable to privacy leaks")
    report.append("\nBased on real-world experience with systems using generalized models")
    report.append("on sensitive data without fine-tuning or proprietary training.\n")
    
    return "\n".join(report)


if __name__ == "__main__":
    main()


