"""
Visualize confusion matrices from attack results.

Usage:
    python visualize_confusion_matrices.py results/face_identification_attack_TIMESTAMP/
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_confusion_matrices(result_dir):
    """Load and visualize confusion matrices."""
    result_path = Path(result_dir)

    # Load confusion matrices
    baseline_cm = np.load(result_path / 'baseline_confusion_matrix.npy')
    attack_cm = np.load(result_path / 'attack_confusion_matrix.npy')

    print(f"Loaded confusion matrices from {result_dir}")
    print(f"Shape: {baseline_cm.shape}")
    print(f"Number of identities: {baseline_cm.shape[0]}")

    # Normalize to show proportions
    baseline_cm_norm = baseline_cm / baseline_cm.sum(axis=1, keepdims=True)
    attack_cm_norm = attack_cm / attack_cm.sum(axis=1, keepdims=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot baseline confusion matrix
    sns.heatmap(baseline_cm_norm, cmap='Blues', ax=ax1,
                cbar_kws={'label': 'Proportion'},
                vmin=0, vmax=1, square=True)
    ax1.set_title('Baseline Confusion Matrix\n(Before Attack)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Predicted Identity', fontsize=13)
    ax1.set_ylabel('True Identity', fontsize=13)

    # Add diagonal accuracy text
    baseline_acc = np.trace(baseline_cm) / baseline_cm.sum()
    ax1.text(0.02, 0.98, f'Accuracy: {baseline_acc*100:.1f}%',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot attack confusion matrix
    sns.heatmap(attack_cm_norm, cmap='Reds', ax=ax2,
                cbar_kws={'label': 'Proportion'},
                vmin=0, vmax=1, square=True)
    ax2.set_title('After Bit Flip Attack\n(Identity Confusion)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Predicted Identity', fontsize=13)
    ax2.set_ylabel('True Identity', fontsize=13)

    # Add diagonal accuracy text
    attack_acc = np.trace(attack_cm) / attack_cm.sum()
    ax2.text(0.02, 0.98, f'Accuracy: {attack_acc*100:.1f}%',
             transform=ax2.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Main title
    fig.suptitle('Identity Confusion Attack: Confusion Matrix Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    output_file = result_path / 'confusion_matrices_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix heatmap to: {output_file}")

    # Also create a difference heatmap
    fig, ax = plt.subplots(figsize=(10, 9))

    # Compute difference (attack - baseline)
    diff_cm = attack_cm_norm - baseline_cm_norm

    # Plot difference
    sns.heatmap(diff_cm, cmap='RdBu_r', ax=ax,
                cbar_kws={'label': 'Change in Proportion'},
                center=0, vmin=-0.5, vmax=0.5, square=True)
    ax.set_title('Identity Confusion Difference\n(Attack - Baseline)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Identity', fontsize=13)
    ax.set_ylabel('True Identity', fontsize=13)

    # Add text with key metrics
    icr_baseline = (baseline_cm.sum() - np.trace(baseline_cm)) / baseline_cm.sum()
    icr_attack = (attack_cm.sum() - np.trace(attack_cm)) / attack_cm.sum()
    icr_increase = (icr_attack - icr_baseline) * 100

    textstr = f'ICR Baseline: {icr_baseline*100:.1f}%\nICR Attack: {icr_attack*100:.1f}%\nIncrease: +{icr_increase:.1f}%'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()

    # Save difference figure
    output_file_diff = result_path / 'confusion_matrices_difference.png'
    plt.savefig(output_file_diff, dpi=300, bbox_inches='tight')
    print(f"✓ Saved difference heatmap to: {output_file_diff}")

    # Print summary statistics
    print("\n" + "="*70)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*70)
    print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
    print(f"Attack Accuracy: {attack_acc*100:.2f}%")
    print(f"Accuracy Drop: {(baseline_acc - attack_acc)*100:.2f}%")
    print()
    print(f"Baseline Identity Confusion Rate: {icr_baseline*100:.2f}%")
    print(f"Attack Identity Confusion Rate: {icr_attack*100:.2f}%")
    print(f"ICR Increase: +{icr_increase:.2f}%")
    print()

    # Find most confused pairs
    print("Top 10 Most Confused Identity Pairs (After Attack):")
    off_diagonal_indices = ~np.eye(attack_cm.shape[0], dtype=bool)
    attack_cm_off_diag = attack_cm.copy()
    attack_cm_off_diag[~off_diagonal_indices] = 0

    # Get top confusion pairs
    flat_indices = np.argsort(attack_cm_off_diag.ravel())[::-1][:10]
    row_indices, col_indices = np.unravel_index(flat_indices, attack_cm.shape)

    for i, (row, col) in enumerate(zip(row_indices, col_indices), 1):
        count = attack_cm[row, col]
        if count > 0:
            print(f"  {i}. Identity {row} → Identity {col}: {int(count)} times ({count/attack_cm[row].sum()*100:.1f}%)")

    print("="*70 + "\n")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_confusion_matrices.py results/face_identification_attack_TIMESTAMP/")
        print("\nAvailable results directories:")
        for d in sorted(Path('results').glob('face_identification_attack_*')):
            print(f"  {d}")
        sys.exit(1)

    result_dir = sys.argv[1]
    visualize_confusion_matrices(result_dir)
