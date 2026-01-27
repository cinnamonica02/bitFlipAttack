"""
CelebA Face Identification Attack - Adapted from V3

Uses CelebA dataset from torchvision for face identification attack.
CelebA advantages:
- Built into torchvision (easy access)
- 202,599 images, 10,177 identities
- High quality face images
- No manual download needed

Key differences from LFW V3:
- Dataset: CelebA (torchvision) instead of LFW (sklearn)
- More identities available (can scale up to 100+)
- Better image quality and diversity
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import FaceNet
from facenet_pytorch import InceptionResnetV1

# Import attack classes
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
import logging
from bitflip_attack.utils.logger import get_attack_logger

logger = get_attack_logger('celeba_identification_attack', level=logging.INFO)


class CelebAIdentificationDataset(Dataset):
    """
    CelebA dataset for face identification.

    CelebA has 10,177 identities but most have very few images.
    We filter to identities with min_faces_per_person images.
    """
    def __init__(self, root='./data', transform=None, min_faces_per_person=20, max_identities=50):
        from torchvision.datasets import CelebA

        self.transform = transform

        print(f"Loading CelebA dataset...")
        print(f"  Root: {root}")
        print(f"  Min faces per person: {min_faces_per_person}")
        print(f"  Max identities: {max_identities}")

        # Download CelebA if needed
        # Note: This downloads ~1.4GB (aligned & cropped version)
        self.celeba = CelebA(
            root=root,
            split='all',  # Use all data (train + valid + test)
            target_type='identity',  # We want identity labels
            download=True
        )

        print(f"✓ Loaded {len(self.celeba)} total images")

        # Count images per identity
        identity_counts = defaultdict(int)
        for idx in range(len(self.celeba)):
            _, identity = self.celeba[idx]
            identity_counts[identity.item()] += 1

        print(f"✓ Found {len(identity_counts)} unique identities")

        # Filter to identities with enough images
        valid_identities = [
            identity for identity, count in identity_counts.items()
            if count >= min_faces_per_person
        ]

        print(f"✓ {len(valid_identities)} identities have ≥{min_faces_per_person} images")

        # Take top N identities by image count
        if max_identities and len(valid_identities) > max_identities:
            top_identities = sorted(
                valid_identities,
                key=lambda x: identity_counts[x],
                reverse=True
            )[:max_identities]
        else:
            top_identities = valid_identities

        top_identity_set = set(top_identities)

        # Build filtered dataset
        self.indices = []
        self.labels = []

        # Create mapping from old identity IDs to new contiguous IDs
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(top_identities))}

        for idx in range(len(self.celeba)):
            _, identity = self.celeba[idx]
            identity_id = identity.item()

            if identity_id in top_identity_set:
                self.indices.append(idx)
                self.labels.append(old_to_new[identity_id])

        self.num_classes = len(top_identities)

        print(f"✓ Final dataset: {len(self.indices)} images across {self.num_classes} identities")
        print(f"✓ Average images per identity: {len(self.indices) / self.num_classes:.1f}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        celeba_idx = self.indices[idx]
        image, _ = self.celeba[celeba_idx]

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return {'image': image, 'label': label}


def create_celeba_dataloaders(root='./data', batch_size=32, img_size=160,
                               min_faces_per_person=20, max_identities=50):
    """Create CelebA dataloaders for face identification."""

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("\n" + "="*70)
    print("Creating CelebA Face Identification Dataset")
    print("="*70)

    dataset = CelebAIdentificationDataset(
        root=root,
        transform=transform,
        min_faces_per_person=min_faces_per_person,
        max_identities=max_identities
    )

    # Split into train/test
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

    # Create class names (CelebA uses numeric IDs)
    class_names = [f"Identity_{i}" for i in range(dataset.num_classes)]

    return train_loader, test_loader, dataset.num_classes, class_names


def load_facenet_embeddings(pretrained='vggface2', device='cuda'):
    """Load FaceNet for embedding extraction."""
    print(f"\nLoading FaceNet (InceptionResnetV1) pretrained on {pretrained}...")
    print("Mode: EMBEDDING EXTRACTION (production-realistic)")

    model = InceptionResnetV1(
        pretrained=pretrained,
        classify=False  # Pure embeddings
    ).to(device).eval()

    print(f"✓ Model loaded on {device}")
    print(f"✓ Output: 512-dim embeddings")

    return model


def create_gallery_embeddings(model, train_loader, device='cuda'):
    """Create gallery of reference embeddings for each identity."""
    print("\n" + "="*70)
    print("Creating Gallery Embeddings (Enrollment Phase)")
    print("="*70)

    model.eval()
    gallery = defaultdict(list)

    with torch.no_grad():
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label']

            embeddings = model(images).cpu()

            for emb, label in zip(embeddings, labels):
                gallery[label.item()].append(emb)

    # Average embeddings per identity
    gallery_embeddings = {}
    for identity_id, emb_list in gallery.items():
        avg_emb = torch.stack(emb_list).mean(dim=0)
        avg_emb = avg_emb / avg_emb.norm()  # L2 normalize
        gallery_embeddings[identity_id] = avg_emb

    print(f"✓ Created gallery for {len(gallery_embeddings)} identities")
    print(f"✓ Embeddings per identity: {np.mean([len(v) for v in gallery.values()]):.1f}")
    print("="*70 + "\n")

    return gallery_embeddings


def evaluate_embedding_based(model, test_loader, gallery_embeddings, device='cuda'):
    """Evaluate using embedding distance matching."""
    model.eval()
    gallery_ids = sorted(gallery_embeddings.keys())
    gallery_matrix = torch.stack([gallery_embeddings[i] for i in gallery_ids]).to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label']

            embeddings = model(images)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

            # Cosine similarity to gallery
            similarities = torch.matmul(embeddings, gallery_matrix.T)

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
    misidentified = (all_preds != all_targets).sum()
    identity_confusion_rate = misidentified / len(all_targets)

    return {
        'accuracy': accuracy,
        'identity_confusion_rate': identity_confusion_rate,
        'confusion_matrix': confusion_matrix,
        'num_classes': num_classes,
        'total_samples': len(all_targets),
        'misidentified_count': int(misidentified)
    }


def print_metrics(metrics, label="Model"):
    """Pretty print metrics."""
    print(f"\n{'='*70}")
    print(f"{label} Evaluation Metrics")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Identity Confusion Rate (ICR): {metrics['identity_confusion_rate']*100:.2f}%")
    print(f"Correctly Identified: {metrics['total_samples'] - metrics['misidentified_count']}/{metrics['total_samples']}")
    print(f"Misidentified: {metrics['misidentified_count']}/{metrics['total_samples']}")
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
    Visualizes specific CelebA identification samples showing 'Before vs After' attack.

    Args:
        model: The attacked model
        sample_data: Dictionary with baseline predictions (from capture_baseline_predictions)
        gallery_embeddings: Gallery embeddings for matching
        class_names: List of identity names/IDs
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

    plt.suptitle('CelebA Face Identification Attack: Before vs After Comparison',
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
    """Run bit flip attack on embedding-based face recognition."""
    print(f"\n{'='*70}")
    print(f"Running Bit Flip Attack (EMBEDDING-BASED)")
    print(f"{'='*70}")
    print(f"Max bit flips: {max_bit_flips}")
    print(f"Bit candidates: {num_candidates}")
    print(f"GA population: {population_size}")
    print(f"GA generations: {generations}")
    print()

    gallery_ids = sorted(gallery_embeddings.keys())
    gallery_matrix = torch.stack([gallery_embeddings[i] for i in gallery_ids]).to(device)

    def embedding_forward_fn(model, batch):
        """Custom forward function for embedding matching."""
        images = batch['image'].to(device)

        embeddings = model(images)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        similarities = torch.matmul(embeddings, gallery_matrix.T)

        # Convert to pseudo-logits
        batch_labels = batch['label']
        pseudo_logits = torch.zeros(len(images), len(gallery_ids), device=device)

        for i, gallery_id in enumerate(gallery_ids):
            pseudo_logits[:, gallery_id] = similarities[:, i]

        return pseudo_logits

    test_dataset = test_loader.dataset

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
        custom_forward_fn=embedding_forward_fn
    )

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

    return attack, results


def save_results(baseline_metrics, attack_metrics, attack_info, output_dir):
    """Save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': 'CelebA',
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
        'root': './data',  # CelebA will be downloaded to ./data/celeba
        'batch_size': 32,
        'img_size': 160,
        'min_faces_per_person': 20,
        'max_identities': 50,  # Start with 50, can increase to 100+
        'max_bit_flips': 10,
        'num_candidates': 500,
        'population_size': 30,
        'generations': 10,
        'pretrained': 'vggface2',
        'num_visualization_samples': 5  # Number of samples for before/after visualization
    }

    print("\n" + "="*70)
    print("CelebA Face Identification Attack")
    print("="*70)
    print("Dataset: CelebA (torchvision)")
    print("Model: FaceNet (InceptionResnetV1)")
    print("Attack: u-μP Bit Flip")
    print("="*70 + "\n")

    print("Note: CelebA download requires 'gdown' package")
    print("If you see download errors, run: pip install gdown\n")

    # Create dataloaders (will download CelebA if needed)
    try:
        train_loader, test_loader, num_classes, class_names = create_celeba_dataloaders(
            root=CONFIG['root'],
            batch_size=CONFIG['batch_size'],
            img_size=CONFIG['img_size'],
            min_faces_per_person=CONFIG['min_faces_per_person'],
            max_identities=CONFIG['max_identities']
        )
    except (RuntimeError, Exception) as e:
        print(f"\n❌ CelebA download failed: {e}")
        print("\nTo fix this:")
        print("  1. Install gdown: pip install gdown")
        print("  2. Run again: python celeba_face_identification_attack.py")
        print("\nAlternatively, use the LFW version that already works:")
        print("  python lfw_face_identification_attack_V3.py")
        return

    # Load FaceNet
    model = load_facenet_embeddings(pretrained=CONFIG['pretrained'], device=device)

    # Create gallery
    gallery_embeddings = create_gallery_embeddings(model, train_loader, device)

    # Baseline evaluation
    print("\n" + "="*70)
    print("BASELINE EVALUATION")
    print("="*70)
    baseline_metrics = evaluate_embedding_based(model, test_loader, gallery_embeddings, device)
    print_metrics(baseline_metrics, label="Baseline")

    # =========================================================================
    # Capture baseline predictions BEFORE attack
    # =========================================================================
    sample_data = capture_baseline_predictions(
        model,
        test_loader,
        gallery_embeddings,
        device=device,
        num_samples=CONFIG['num_visualization_samples']
    )

    # Run attack
    attack, results = run_bit_flip_attack_embedding_based(
        model, test_loader, gallery_embeddings,
        max_bit_flips=CONFIG['max_bit_flips'],
        num_candidates=CONFIG['num_candidates'],
        population_size=CONFIG['population_size'],
        generations=CONFIG['generations'],
        device=device
    )

    # Post-attack evaluation
    print("\n" + "="*70)
    print("POST-ATTACK EVALUATION")
    print("="*70)
    attack_metrics = evaluate_embedding_based(model, test_loader, gallery_embeddings, device)
    print_metrics(attack_metrics, label="After Attack")

    # Save results
    output_dir = f"results/celeba_identification_attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    attack_info = {
        'bits_flipped': attack.bits_flipped,
        'config': CONFIG
    }
    save_results(baseline_metrics, attack_metrics, attack_info, output_dir)

    # =========================================================================
    # Generate sample visualization showing before/after
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
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print(f"  • results.json - Full metrics")
    print(f"  • baseline_confusion_matrix.npy - Baseline confusion matrix")
    print(f"  • attack_confusion_matrix.npy - Attack confusion matrix")
    print(f"  • identification_samples_comparison.png - Before/After visualization")
    print(f"\nKey Findings:")
    print(f"  • Dataset: CelebA ({num_classes} identities)")
    print(f"  • Baseline ICR: {baseline_metrics['identity_confusion_rate']*100:.2f}%")
    print(f"  • Attack ICR: {attack_metrics['identity_confusion_rate']*100:.2f}%")
    print(f"  • ICR Increase: +{(attack_metrics['identity_confusion_rate'] - baseline_metrics['identity_confusion_rate'])*100:.2f}%")
    print(f"  • Bits Used: {attack.bits_flipped}")
    print(f"  • Bit Efficiency: {(attack_metrics['identity_confusion_rate'] / max(attack.bits_flipped, 1))*100:.2f}% ICR per bit")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
