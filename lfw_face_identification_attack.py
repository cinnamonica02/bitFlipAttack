"""
Face Identification Attack using Bit Flips on FaceNet (InceptionResnetV1)

This script demonstrates identity confusion attacks where Person_A is misidentified as Person_B.

Attack Framing: "False Identity Injection Attack" or "Identity Confusion Attack"
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FaceNet (installed via requirements.txt from GitHub)
from facenet_pytorch import InceptionResnetV1, MTCNN

# Import attack classes
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack
import logging
from bitflip_attack.utils.logger import get_attack_logger

logger = get_attack_logger('lfw_face_identification_attack', level=logging.INFO)


class LFWIdentificationDataset(Dataset):
    """
    LFW dataset for face identification (multi-class classification).

    Each person is a separate class. We use a subset of identities with sufficient images.
    """
    def __init__(self, transform=None, min_faces_per_person=20, max_identities=50, resize=0.5):
        self.transform = transform

        print(f"Loading LFW dataset for identification...")
        print(f"  Min faces per person: {min_faces_per_person}")
        print(f"  Max identities: {max_identities}")

        # Fetch LFW people dataset
        lfw_data = fetch_lfw_people(
            data_home='./data',
            min_faces_per_person=min_faces_per_person,
            resize=resize,
            color=True
        )

        self.images = lfw_data.images  # Shape: (n_samples, H, W, 3)
        self.targets = lfw_data.target  # Person IDs (0 to n_classes-1)
        self.target_names = lfw_data.target_names  # Person names

        # Limit to max_identities if specified
        if max_identities and len(self.target_names) > max_identities:
            # Select identities with most images
            identity_counts = defaultdict(int)
            for target in self.targets:
                identity_counts[target] += 1

            top_identities = sorted(identity_counts.items(), key=lambda x: x[1], reverse=True)[:max_identities]
            top_identity_ids = set([identity for identity, count in top_identities])

            # Filter dataset
            mask = np.isin(self.targets, list(top_identity_ids))
            self.images = self.images[mask]
            old_targets = self.targets[mask]

            # Remap targets to 0-based indices
            old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(top_identity_ids))}
            self.targets = np.array([old_to_new[old_id] for old_id in old_targets])
            self.target_names = [self.target_names[old_id] for old_id in sorted(top_identity_ids)]

        self.num_classes = len(self.target_names)

        print(f"✓ Loaded {len(self.images)} images")
        print(f"✓ Number of identities: {self.num_classes}")
        print(f"✓ Average images per identity: {len(self.images) / self.num_classes:.1f}")

        # Print identity distribution
        unique, counts = np.unique(self.targets, return_counts=True)
        print(f"✓ Identity distribution: min={counts.min()}, max={counts.max()}, median={np.median(counts):.0f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # LFW images are float in [0, 1], convert to uint8
        image = (self.images[idx] * 255).astype(np.uint8)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = int(self.targets[idx])

        return {'image': image, 'label': label}


def create_identification_dataloaders(batch_size=32, img_size=160,
                                     min_faces_per_person=20, max_identities=50):
    """
    Create train/test dataloaders for face identification.

    Args:
        batch_size: Batch size for training
        img_size: Image size (FaceNet expects 160x160)
        min_faces_per_person: Minimum images per identity
        max_identities: Maximum number of identities to include

    Returns:
        train_loader, test_loader, num_classes, class_names
    """
    # FaceNet expects 160x160 images with specific normalization
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("\n" + "="*70)
    print("Creating Face Identification Dataset")
    print("="*70)

    dataset = LFWIdentificationDataset(
        transform=transform,
        min_faces_per_person=min_faces_per_person,
        max_identities=max_identities
    )

    # Split into train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\n✓ Train set: {train_size} samples")
    print(f"✓ Test set: {test_size} samples")
    print("="*70 + "\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    return train_loader, test_loader, dataset.num_classes, dataset.target_names


def load_facenet_classifier(num_classes, pretrained='vggface2', device='cuda'):
    """
    Load FaceNet (InceptionResnetV1) as a classifier.

    We use the pretrained model and replace the final layer for our num_classes.
    """
    print(f"\nLoading FaceNet (InceptionResnetV1) pretrained on {pretrained}...")

    # Load pretrained model
    model = InceptionResnetV1(
        pretrained=pretrained,
        classify=False,  # We'll add our own classifier
        num_classes=None
    ).to(device)

    # Add classification head for our identities
    model.logits = nn.Linear(512, num_classes).to(device)
    model.classify = True

    print(f"✓ Model loaded on {device}")
    print(f"✓ Pretrained on: {pretrained}")
    print(f"✓ Embedding dimension: 512")
    print(f"✓ Output classes: {num_classes}")

    return model


def fine_tune_facenet(model, train_loader, test_loader, epochs=10, device='cuda', lr=0.001):
    """
    Fine-tune the FaceNet classifier on LFW identities.

    We only train the final classification layer to adapt to LFW identities.
    """
    print("\n" + "="*70)
    print("Fine-tuning FaceNet on LFW Identities")
    print("="*70)

    # Freeze all layers except the final classifier
    for name, param in model.named_parameters():
        if 'logits' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs, targets = batch['image'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch['image'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {100*train_acc:.2f}%')
        print(f'  Val Acc: {100*val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc

        scheduler.step()

    print(f"\n✓ Fine-tuning complete. Best accuracy: {100*best_acc:.2f}%")
    print("="*70 + "\n")

    # Unfreeze all layers for attack
    for param in model.parameters():
        param.requires_grad = True

    return model, best_acc


def evaluate_identity_confusion(model, test_loader, device='cuda'):
    """
    Comprehensive evaluation including identity confusion metrics.

    Returns detailed metrics for paper including confusion matrix.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    num_classes = len(np.unique(all_targets))
    confusion_matrix = np.zeros((num_classes, num_classes))

    for true_label, pred_label in zip(all_targets, all_preds):
        confusion_matrix[true_label, pred_label] += 1

    # Overall accuracy
    accuracy = (all_preds == all_targets).mean()

    # Identity confusion rate (% of misidentifications)
    total_samples = len(all_targets)
    misidentified = (all_preds != all_targets).sum()
    identity_confusion_rate = misidentified / total_samples

    # Find most confused identity pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion_matrix[i, j] > 0:
                confused_pairs.append((i, j, int(confusion_matrix[i, j])))

    confused_pairs = sorted(confused_pairs, key=lambda x: x[2], reverse=True)[:10]

    metrics = {
        'accuracy': accuracy,
        'identity_confusion_rate': identity_confusion_rate,
        'confusion_matrix': confusion_matrix,
        'most_confused_pairs': confused_pairs,
        'num_classes': num_classes,
        'total_samples': total_samples,
        'misidentified_count': int(misidentified)
    }

    return metrics


def print_metrics(metrics, label="Model"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*70}")
    print(f"{label} Evaluation Metrics")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Identity Confusion Rate (ICR): {metrics['identity_confusion_rate']*100:.2f}%")
    print(f"Correctly Identified: {metrics['total_samples'] - metrics['misidentified_count']}/{metrics['total_samples']}")
    print(f"Misidentified: {metrics['misidentified_count']}/{metrics['total_samples']}")
    print(f"\nTop 5 Most Confused Identity Pairs:")
    for i, (source, target, count) in enumerate(metrics['most_confused_pairs'][:5], 1):
        print(f"  {i}. Person {source} → Person {target}: {count} times")
    print(f"{'='*70}\n")


def run_bit_flip_attack(model, test_loader, max_bit_flips=10, num_candidates=500,
                       population_size=30, generations=10, device='cuda',
                       attack_mode='untargeted', target_class=None):
    """
    Run bit flip attack on face identification model.

    Args:
        model: FaceNet classifier
        test_loader: Test dataloader
        max_bit_flips: Maximum number of bits to flip
        num_candidates: Number of bit candidates to consider
        population_size: Genetic algorithm population size
        generations: Genetic algorithm generations
        device: cuda/cpu
        attack_mode: 'targeted' or 'untargeted'
        target_class: Target identity for targeted attacks

    Returns:
        attack_results: Dictionary with attack metrics
    """
    print(f"\n{'='*70}")
    print(f"Running Bit Flip Attack ({attack_mode.upper()})")
    print(f"{'='*70}")
    print(f"Max bit flips: {max_bit_flips}")
    print(f"Bit candidates: {num_candidates}")
    print(f"GA population: {population_size}")
    print(f"GA generations: {generations}")
    if attack_mode == 'targeted':
        print(f"Target class: {target_class}")
    print()

    # Create dataset wrapper for attack
    test_dataset = test_loader.dataset

    # Custom forward function for FaceNet
    def facenet_forward(model, batch):
        inputs = batch['image'].to(device)
        return model(inputs)

    # Initialize attack
    attack = UmupBitFlipAttack(
        model=model,
        dataset=test_dataset,
        target_asr=0.5,  # Target 50% attack success rate
        max_bit_flips=max_bit_flips,
        accuracy_threshold=0.10,  # Allow 10% accuracy drop
        device=device,
        attack_mode=attack_mode,
        layer_sensitivity=True,
        hybrid_sensitivity=True,
        alpha=0.5,
        custom_forward_fn=facenet_forward
    )

    # Run attack
    results = attack.perform_attack(
        target_class=target_class,
        num_candidates=num_candidates,
        population_size=population_size,
        generations=generations
    )

    print(f"\n✓ Attack completed!")
    print(f"  Bits flipped: {attack.bits_flipped}")
    print(f"  Original accuracy: {attack.original_accuracy*100:.2f}%")
    print(f"  Final accuracy: {attack.final_accuracy*100:.2f}%")
    print(f"  Accuracy drop: {(attack.original_accuracy - attack.final_accuracy)*100:.2f}%")
    print(f"  Initial ASR: {attack.initial_asr*100:.2f}%")
    print(f"  Final ASR: {attack.final_asr*100:.2f}%")
    print(f"  ASR improvement: {(attack.final_asr - attack.initial_asr)*100:.2f}%")

    return attack, results


def save_results(baseline_metrics, attack_metrics, attack_info, output_dir):
    """Save results for paper."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    results = {
        'baseline': {
            'accuracy': float(baseline_metrics['accuracy']),
            'identity_confusion_rate': float(baseline_metrics['identity_confusion_rate']),
            'misidentified_count': int(baseline_metrics['misidentified_count']),
            'total_samples': int(baseline_metrics['total_samples'])
        },
        'attack': {
            'accuracy': float(attack_metrics['accuracy']),
            'identity_confusion_rate': float(attack_metrics['identity_confusion_rate']),
            'misidentified_count': int(attack_metrics['misidentified_count']),
            'total_samples': int(attack_metrics['total_samples']),
            'bits_flipped': attack_info['bits_flipped'],
            'bit_efficiency': float(attack_metrics['identity_confusion_rate']) / max(attack_info['bits_flipped'], 1)
        },
        'improvement': {
            'accuracy_drop': float(baseline_metrics['accuracy'] - attack_metrics['accuracy']),
            'icr_increase': float(attack_metrics['identity_confusion_rate'] - baseline_metrics['identity_confusion_rate'])
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save confusion matrices
    np.save(output_dir / 'baseline_confusion_matrix.npy', baseline_metrics['confusion_matrix'])
    np.save(output_dir / 'attack_confusion_matrix.npy', attack_metrics['confusion_matrix'])

    print(f"\n✓ Results saved to {output_dir}/")


def plot_comparison(baseline_metrics, attack_metrics, attack_info, output_dir):
    """Generate comparison plots for paper."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Accuracy and ICR comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    categories = ['Baseline', f'After Attack\n({attack_info["bits_flipped"]} bits)']
    accuracies = [baseline_metrics['accuracy'] * 100, attack_metrics['accuracy'] * 100]
    icrs = [baseline_metrics['identity_confusion_rate'] * 100, attack_metrics['identity_confusion_rate'] * 100]

    ax1.bar(categories, accuracies, color=['green', 'red'], alpha=0.7)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    ax2.bar(categories, icrs, color=['green', 'red'], alpha=0.7)
    ax2.set_ylabel('Identity Confusion Rate (%)', fontsize=12)
    ax2.set_title('Identity Confusion Rate (Attack Success)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(icrs) * 1.2])
    for i, v in enumerate(icrs):
        ax2.text(i, v + max(icrs)*0.02, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison plot to {output_dir}/comparison_metrics.png")


def main():
    """Main execution function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration
    CONFIG = {
        'batch_size': 32,
        'img_size': 160,
        'min_faces_per_person': 20,
        'max_identities': 50,  # Use 50 identities for faster experiments
        'fine_tune_epochs': 10,
        'fine_tune_lr': 0.001,
        'max_bit_flips': 10,
        'num_candidates': 500,
        'population_size': 30,
        'generations': 10,
        'pretrained': 'vggface2'
    }

    # Create dataloaders
    train_loader, test_loader, num_classes, class_names = create_identification_dataloaders(
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size'],
        min_faces_per_person=CONFIG['min_faces_per_person'],
        max_identities=CONFIG['max_identities']
    )

    # Load FaceNet classifier
    model = load_facenet_classifier(num_classes, pretrained=CONFIG['pretrained'], device=device)

    # Fine-tune on LFW identities
    model, train_acc = fine_tune_facenet(
        model, train_loader, test_loader,
        epochs=CONFIG['fine_tune_epochs'],
        device=device,
        lr=CONFIG['fine_tune_lr']
    )

    # Evaluate baseline
    print("\n" + "="*70)
    print("BASELINE EVALUATION")
    print("="*70)
    baseline_metrics = evaluate_identity_confusion(model, test_loader, device)
    print_metrics(baseline_metrics, label="Baseline")

    # Run bit flip attack (untargeted)
    attack, results = run_bit_flip_attack(
        model, test_loader,
        max_bit_flips=CONFIG['max_bit_flips'],
        num_candidates=CONFIG['num_candidates'],
        population_size=CONFIG['population_size'],
        generations=CONFIG['generations'],
        device=device,
        attack_mode='untargeted'
    )

    # Evaluate after attack
    print("\n" + "="*70)
    print("POST-ATTACK EVALUATION")
    print("="*70)
    attack_metrics = evaluate_identity_confusion(model, test_loader, device)
    print_metrics(attack_metrics, label="After Attack")

    # Save results
    output_dir = f"results/face_identification_attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    attack_info = {
        'bits_flipped': attack.bits_flipped,
        'config': CONFIG
    }
    save_results(baseline_metrics, attack_metrics, attack_info, output_dir)
    plot_comparison(baseline_metrics, attack_metrics, attack_info, output_dir)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print(f"\nKey Findings:")
    print(f"  • Baseline ICR: {baseline_metrics['identity_confusion_rate']*100:.2f}%")
    print(f"  • Attack ICR: {attack_metrics['identity_confusion_rate']*100:.2f}%")
    print(f"  • ICR Increase: +{(attack_metrics['identity_confusion_rate'] - baseline_metrics['identity_confusion_rate'])*100:.2f}%")
    print(f"  • Bits Used: {attack.bits_flipped}")
    print(f"  • Bit Efficiency: {(attack_metrics['identity_confusion_rate'] / max(attack.bits_flipped, 1))*100:.2f}% ICR per bit")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
