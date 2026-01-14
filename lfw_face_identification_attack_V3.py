"""
Face Identification Attack V3 - With Sample Visualization (Before/After Attack)

This version extends V2 by adding visual comparison of specific face samples
showing predictions before and after the bit flip attack.

Key Features:
- V2: Embedding-based matching (production-realistic)
- V3: + Sample visualization showing before/after attack predictions
- Captures baseline predictions BEFORE attack
- Shows side-by-side comparison of specific misidentified faces
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Headless mode for remote environments
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FaceNet
from facenet_pytorch import InceptionResnetV1, MTCNN

# Import attack classes
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
import logging
from bitflip_attack.utils.logger import get_attack_logger

logger = get_attack_logger('lfw_identification_attack_v3', level=logging.INFO)


class LFWIdentificationDataset(Dataset):
    """LFW dataset for face identification."""
    def __init__(self, transform=None, min_faces_per_person=20, max_identities=50, resize=0.5):
        self.transform = transform

        print(f"Loading LFW dataset for identification...")
        print(f"  Min faces per person: {min_faces_per_person}")
        print(f"  Max identities: {max_identities}")

        lfw_data = fetch_lfw_people(
            data_home='./data',
            min_faces_per_person=min_faces_per_person,
            resize=resize,
            color=True
        )

        self.images = lfw_data.images
        self.targets = lfw_data.target
        self.target_names = lfw_data.target_names

        # Limit to max_identities
        if max_identities and len(self.target_names) > max_identities:
            identity_counts = defaultdict(int)
            for target in self.targets:
                identity_counts[target] += 1

            top_identities = sorted(identity_counts.items(), key=lambda x: x[1], reverse=True)[:max_identities]
            top_identity_ids = set([identity for identity, count in top_identities])

            mask = np.isin(self.targets, list(top_identity_ids))
            self.images = self.images[mask]
            old_targets = self.targets[mask]

            old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(top_identity_ids))}
            self.targets = np.array([old_to_new[old_id] for old_id in old_targets])
            self.target_names = [self.target_names[old_id] for old_id in sorted(top_identity_ids)]

        self.num_classes = len(self.target_names)

        print(f"✓ Loaded {len(self.images)} images")
        print(f"✓ Number of identities: {self.num_classes}")
        print(f"✓ Average images per identity: {len(self.images) / self.num_classes:.1f}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = (self.images[idx] * 255).astype(np.uint8)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = int(self.targets[idx])
        return {'image': image, 'label': label}


def create_identification_dataloaders(batch_size=32, img_size=160,
                                     min_faces_per_person=20, max_identities=50):
    """Create dataloaders."""
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


def load_facenet_embeddings(pretrained='vggface2', device='cuda'):
    """
    Load FaceNet for embedding extraction (NO classification head).

    This is the production-realistic approach!
    """
    print(f"\nLoading FaceNet (InceptionResnetV1) pretrained on {pretrained}...")
    print("Mode: EMBEDDING EXTRACTION (production-realistic)")

    model = InceptionResnetV1(
        pretrained=pretrained,
        classify=False  # No classification head! Pure embeddings
    ).to(device).eval()

    print(f"✓ Model loaded on {device}")
    print(f"✓ Pretrained on: {pretrained}")
    print(f"✓ Output: 512-dim embeddings (NOT class logits)")
    print(f"✓ Matching: Distance-based (cosine similarity)")

    return model


def create_gallery_embeddings(model, train_loader, device='cuda'):
    """
    Create a gallery of reference embeddings for each identity.

    This is like an "enrollment" phase in face recognition systems.
    We store one representative embedding per person.
    """
    print("\n" + "="*70)
    print("Creating Gallery Embeddings (Enrollment Phase)")
    print("="*70)

    model.eval()
    gallery = defaultdict(list)

    with torch.no_grad():
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label']

            # Extract embeddings
            embeddings = model(images).cpu()

            # Store by identity
            for emb, label in zip(embeddings, labels):
                gallery[label.item()].append(emb)

    # Average embeddings per identity (more robust)
    gallery_embeddings = {}
    for identity_id, emb_list in gallery.items():
        avg_emb = torch.stack(emb_list).mean(dim=0)
        # L2 normalize
        avg_emb = avg_emb / avg_emb.norm()
        gallery_embeddings[identity_id] = avg_emb

    print(f"✓ Created gallery for {len(gallery_embeddings)} identities")
    print(f"✓ Embeddings per identity: {np.mean([len(v) for v in gallery.values()]):.1f}")
    print("="*70 + "\n")

    return gallery_embeddings


def evaluate_embedding_based(model, test_loader, gallery_embeddings, device='cuda'):
    """
    Evaluate using embedding distance matching (production approach).

    For each test image:
    1. Extract embedding
    2. Compare to all gallery embeddings
    3. Predict identity with highest similarity
    """
    model.eval()
    gallery_ids = sorted(gallery_embeddings.keys())
    gallery_matrix = torch.stack([gallery_embeddings[i] for i in gallery_ids]).to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label']

            # Extract embeddings
            embeddings = model(images)
            # L2 normalize
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

            # Compute cosine similarity to gallery
            similarities = torch.matmul(embeddings, gallery_matrix.T)  # (batch, num_identities)

            # Predict identity with highest similarity
            _, predicted_indices = similarities.max(dim=1)
            predicted_ids = [gallery_ids[idx] for idx in predicted_indices.cpu().numpy()]

            all_preds.extend(predicted_ids)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    num_classes = len(gallery_ids)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for true_label, pred_label in zip(all_targets, all_preds):
        confusion_matrix[true_label, pred_label] += 1

    accuracy = (all_preds == all_targets).mean()
    total_samples = len(all_targets)
    misidentified = (all_preds != all_targets).sum()
    identity_confusion_rate = misidentified / total_samples

    # Find most confused pairs
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
    """Pretty print metrics."""
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


def capture_baseline_predictions(model, test_loader, gallery_embeddings, device='cuda', num_samples=10):
    """
    Capture baseline predictions on sample images BEFORE attack.

    This is crucial - we need to save predictions before the model is attacked!

    Returns:
        sample_data: Dict with images, true labels, baseline predictions, and similarities
    """
    print(f"\n{'='*70}")
    print(f"Capturing Baseline Predictions for {num_samples} Sample Images")
    print(f"{'='*70}")

    model.eval()
    gallery_ids = sorted(gallery_embeddings.keys())
    gallery_matrix = torch.stack([gallery_embeddings[i] for i in gallery_ids]).to(device)

    # Get a batch from test loader
    batch = next(iter(test_loader))
    images = batch['image'][:num_samples].to(device)
    true_labels = batch['label'][:num_samples].cpu().numpy()

    with torch.no_grad():
        # Extract embeddings
        embeddings = model(images)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(embeddings, gallery_matrix.T)

        # Get predictions
        _, predicted_indices = similarities.max(dim=1)
        baseline_preds = [gallery_ids[idx] for idx in predicted_indices.cpu().numpy()]
        baseline_sims = similarities.max(dim=1)[0].cpu().numpy()

    print(f"✓ Captured baseline predictions for {num_samples} samples")
    print(f"  Baseline accuracy on samples: {sum(bp == tl for bp, tl in zip(baseline_preds, true_labels))}/{num_samples}")
    print(f"{'='*70}\n")

    return {
        'images': images.cpu(),
        'true_labels': true_labels,
        'baseline_preds': baseline_preds,
        'baseline_sims': baseline_sims
    }


def plot_attack_comparison_samples(model, sample_data, gallery_embeddings,
                                   class_names, output_dir, device='cuda'):
    """
    Visualizes specific LFW identification samples showing 'Before vs After' attack.

    Args:
        model: The attacked model
        sample_data: Dictionary with baseline predictions (from capture_baseline_predictions)
        gallery_embeddings: Gallery embeddings for matching
        class_names: List of person names
        output_dir: Directory to save the visualization
        device: cuda/cpu
    """
    print(f"\n{'='*70}")
    print("Generating Attack Comparison Sample Visualization")
    print(f"{'='*70}")

    gallery_ids = sorted(gallery_embeddings.keys())
    gallery_matrix = torch.stack([gallery_embeddings[i] for i in gallery_ids]).to(device)

    images = sample_data['images'].to(device)
    true_labels = sample_data['true_labels']
    baseline_preds = sample_data['baseline_preds']
    baseline_sims = sample_data['baseline_sims']

    num_samples = len(images)

    # Get POST-ATTACK predictions
    model.eval()
    with torch.no_grad():
        embeddings = model(images)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        similarities = torch.matmul(embeddings, gallery_matrix.T)
        _, predicted_indices = similarities.max(dim=1)
        attack_preds = [gallery_ids[idx] for idx in predicted_indices.cpu().numpy()]
        attack_sims = similarities.max(dim=1)[0].cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Unnormalize image
        img_np = images[i].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)

        true_label = true_labels[i]
        baseline_pred = baseline_preds[i]
        attack_pred = attack_preds[i]

        # Column 1: Original Image with True Label
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"TRUE IDENTITY\n{class_names[true_label]}",
                            fontsize=11, fontweight='bold', color='darkblue')
        axes[i, 0].axis('off')

        # Column 2: Before Attack Result
        baseline_correct = baseline_pred == true_label
        baseline_color = 'lightgreen' if baseline_correct else 'lightcoral'
        baseline_status = '✓ Correct' if baseline_correct else '✗ Misidentified'

        baseline_text = (
            f"BEFORE ATTACK\n\n"
            f"Predicted:\n{class_names[baseline_pred]}\n\n"
            f"Confidence: {baseline_sims[i]:.4f}\n\n"
            f"Status: {baseline_status}"
        )

        axes[i, 1].text(0.5, 0.5, baseline_text, ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=baseline_color, alpha=0.7, edgecolor='black', linewidth=2))
        axes[i, 1].set_title("Baseline Model", fontsize=11, fontweight='bold')
        axes[i, 1].axis('off')

        # Column 3: After Attack Result
        attack_correct = attack_pred == true_label
        attack_color = 'lightgreen' if attack_correct else 'salmon'
        attack_status = '✓ Correct' if attack_correct else '✗ MISIDENTIFIED'

        # Check if attack caused a change
        attack_changed = baseline_pred != attack_pred
        change_indicator = " 🔥 [CHANGED!]" if attack_changed else ""

        attack_text = (
            f"AFTER ATTACK{change_indicator}\n\n"
            f"Predicted:\n{class_names[attack_pred]}\n\n"
            f"Confidence: {attack_sims[i]:.4f}\n\n"
            f"Status: {attack_status}"
        )

        axes[i, 2].text(0.5, 0.5, attack_text, ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=attack_color, alpha=0.7, edgecolor='red', linewidth=3))
        axes[i, 2].set_title("Attacked Model", fontsize=11, fontweight='bold', color='red')
        axes[i, 2].axis('off')

    plt.suptitle('Face Identification Attack: Before vs After Comparison',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = Path(output_dir) / 'identification_samples_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved sample visualization to {save_path}")
    print(f"  Samples shown: {num_samples}")
    print(f"  Baseline correct: {sum(bp == tl for bp, tl in zip(baseline_preds, true_labels))}/{num_samples}")
    print(f"  Attack correct: {sum(ap == tl for ap, tl in zip(attack_preds, true_labels))}/{num_samples}")
    print(f"  Predictions changed: {sum(bp != ap for bp, ap in zip(baseline_preds, attack_preds))}/{num_samples}")
    print(f"{'='*70}\n")


def run_bit_flip_attack_embedding_based(model, test_loader, gallery_embeddings,
                                        max_bit_flips=10, num_candidates=500,
                                        population_size=30, generations=10, device='cuda'):
    """
    Run bit flip attack on embedding-based face recognition.

    KEY: We use a custom forward function that converts embeddings to "pseudo-logits"
    so the attack infrastructure can still use classification-style evaluation!
    """
    print(f"\n{'='*70}")
    print(f"Running Bit Flip Attack (EMBEDDING-BASED)")
    print(f"{'='*70}")
    print(f"Max bit flips: {max_bit_flips}")
    print(f"Bit candidates: {num_candidates}")
    print(f"GA population: {population_size}")
    print(f"GA generations: {generations}")
    print()

    # Prepare gallery for fast lookup
    gallery_ids = sorted(gallery_embeddings.keys())
    gallery_matrix = torch.stack([gallery_embeddings[i] for i in gallery_ids]).to(device)

    def embedding_forward_fn(model, batch):
        """
        Custom forward function that does embedding matching.

        Returns "pseudo-logits" based on cosine similarity.
        This tricks the attack infrastructure into thinking it's classification!
        """
        images = batch['image'].to(device)

        # Extract embeddings
        embeddings = model(images)
        # L2 normalize
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        # Compute cosine similarity to gallery (acts as "logits")
        similarities = torch.matmul(embeddings, gallery_matrix.T)  # (batch, num_identities)

        # Remap to actual class indices
        # Create a mapping from gallery position to actual class ID
        batch_labels = batch['label']
        pseudo_logits = torch.zeros(len(images), len(gallery_ids), device=device)

        for i, gallery_id in enumerate(gallery_ids):
            pseudo_logits[:, gallery_id] = similarities[:, i]

        return pseudo_logits

    # Create dataset wrapper
    test_dataset = test_loader.dataset

    # Initialize attack
    attack = UmupBitFlipAttack(
        model=model,
        dataset=test_dataset,
        target_asr=0.5,
        max_bit_flips=max_bit_flips,
        accuracy_threshold=0.10,
        device=device,
        attack_mode='untargeted',
        layer_sensitivity=True,
        hybrid_sensitivity=True,
        alpha=0.5,
        custom_forward_fn=embedding_forward_fn  # Our embedding-based evaluation!
    )

    # Run attack
    results = attack.perform_attack(
        target_class=None,
        num_candidates=num_candidates,
        population_size=population_size,
        generations=generations
    )

    print(f"\n✓ Attack completed!")
    print(f"  Bits flipped: {attack.bits_flipped}")
    print(f"  Original accuracy: {attack.original_accuracy*100:.2f}%")
    print(f"  Final accuracy: {attack.final_accuracy*100:.2f}%")
    print(f"  Accuracy drop: {(attack.original_accuracy - attack.final_accuracy)*100:.2f}%")

    return attack, results


def save_results(baseline_metrics, attack_metrics, attack_info, output_dir):
    """Save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'version': 'V3_with_sample_visualization',
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

    np.save(output_dir / 'baseline_confusion_matrix.npy', baseline_metrics['confusion_matrix'])
    np.save(output_dir / 'attack_confusion_matrix.npy', attack_metrics['confusion_matrix'])

    print(f"\n✓ Results saved to {output_dir}/")


def main():
    """Main execution function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    CONFIG = {
        'batch_size': 32,
        'img_size': 160,
        'min_faces_per_person': 20,
        'max_identities': 50,
        'max_bit_flips': 10,
        'num_candidates': 500,
        'population_size': 30,
        'generations': 10,
        'pretrained': 'vggface2',
        'num_visualization_samples': 5  # NEW: Number of samples to visualize
    }

    # Create dataloaders
    train_loader, test_loader, num_classes, class_names = create_identification_dataloaders(
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size'],
        min_faces_per_person=CONFIG['min_faces_per_person'],
        max_identities=CONFIG['max_identities']
    )

    # Load FaceNet (embedding mode - NO classification head!)
    model = load_facenet_embeddings(pretrained=CONFIG['pretrained'], device=device)

    # Create gallery (enrollment phase)
    gallery_embeddings = create_gallery_embeddings(model, train_loader, device)

    # Evaluate baseline
    print("\n" + "="*70)
    print("BASELINE EVALUATION (Embedding-Based Matching)")
    print("="*70)
    baseline_metrics = evaluate_embedding_based(model, test_loader, gallery_embeddings, device)
    print_metrics(baseline_metrics, label="Baseline (V3)")

    # =========================================================================
    # NEW in V3: Capture baseline predictions BEFORE attack
    # =========================================================================
    sample_data = capture_baseline_predictions(
        model,
        test_loader,
        gallery_embeddings,
        device=device,
        num_samples=CONFIG['num_visualization_samples']
    )

    # Run bit flip attack
    attack, results = run_bit_flip_attack_embedding_based(
        model, test_loader, gallery_embeddings,
        max_bit_flips=CONFIG['max_bit_flips'],
        num_candidates=CONFIG['num_candidates'],
        population_size=CONFIG['population_size'],
        generations=CONFIG['generations'],
        device=device
    )

    # Evaluate after attack
    print("\n" + "="*70)
    print("POST-ATTACK EVALUATION (Embedding-Based Matching)")
    print("="*70)
    attack_metrics = evaluate_embedding_based(model, test_loader, gallery_embeddings, device)
    print_metrics(attack_metrics, label="After Attack (V3)")

    # Save results
    output_dir = f"results/face_identification_attack_V3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    attack_info = {
        'bits_flipped': attack.bits_flipped,
        'config': CONFIG
    }
    save_results(baseline_metrics, attack_metrics, attack_info, output_dir)

    # =========================================================================
    # NEW in V3: Generate sample visualization showing before/after
    # =========================================================================
    plot_attack_comparison_samples(
        model=model,
        sample_data=sample_data,
        gallery_embeddings=gallery_embeddings,
        class_names=class_names,
        output_dir=output_dir,
        device=device
    )

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE! (V3 - With Sample Visualization)")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print(f"\nKey Findings (V3):")
    print(f"  • Approach: Distance-based matching (production-realistic)")
    print(f"  • Baseline ICR: {baseline_metrics['identity_confusion_rate']*100:.2f}%")
    print(f"  • Attack ICR: {attack_metrics['identity_confusion_rate']*100:.2f}%")
    print(f"  • ICR Increase: +{(attack_metrics['identity_confusion_rate'] - baseline_metrics['identity_confusion_rate'])*100:.2f}%")
    print(f"  • Bits Used: {attack.bits_flipped}")
    print(f"  • Bit Efficiency: {(attack_metrics['identity_confusion_rate'] / max(attack.bits_flipped, 1))*100:.2f}% ICR per bit")
    print(f"  • Sample visualization saved: identification_samples_comparison.png")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
