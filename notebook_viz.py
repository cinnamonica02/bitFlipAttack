#!/usr/bin/env python3
"""
Standalone Visualization for Face Identification Attack Results
==============================================================
Optimized for SSH/Remote environments (RunPod).
Saves all plots as PNG files in the results directory.
"""

import os
import sys
import json
import numpy as np
import matplotlib
# Force Matplotlib to use 'Agg' backend for Headless environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: Configuration
# =============================================================================

# UPDATE THIS TO YOUR SPECIFIC RESULTS FOLDER
RESULTS_DIR = Path("results/face_identification_attack_V2_20260114_091028")
NOTEBOOK_VIZ_DIR = RESULTS_DIR / "notebook_visualizations"
NOTEBOOK_VIZ_DIR.mkdir(exist_ok=True, parents=True)

# Set global plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# =============================================================================
# SECTION 2: Load Data
# =============================================================================

print("\n" + "="*80)
print(f"LOADING RESULTS FROM: {RESULTS_DIR}")
print("="*80)

try:
    with open(RESULTS_DIR / 'results.json', 'r') as f:
        results = json.load(f)
    baseline_cm = np.load(RESULTS_DIR / 'baseline_confusion_matrix.npy')
    attack_cm = np.load(RESULTS_DIR / 'attack_confusion_matrix.npy')
except FileNotFoundError as e:
    print(f"❌ Error: Could not find result files. Please check the path: {RESULTS_DIR}")
    sys.exit(1)

# Extract key metrics
baseline_acc = results['baseline']['accuracy']
attack_acc = results['attack']['accuracy']
baseline_icr = results['baseline']['identity_confusion_rate']
attack_icr = results['attack']['identity_confusion_rate']
bits_flipped = results['attack']['bits_flipped']
total_samples = results['baseline']['total_samples']

print(f"✅ Data Loaded.")
print(f"   Baseline Acc: {baseline_acc*100:.2f}% | Attack Acc: {attack_acc*100:.2f}%")

# =============================================================================
# SECTION 3: Visualizations
# =============================================================================

# -----------------------------------------------------------------------------
# VIZ 1: Detailed Confusion Analysis
# -----------------------------------------------------------------------------

def plot_detailed_confusion_comparison():
    """Plot confusion matrices with detailed per-class analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # Normalize confusion matrices
    baseline_cm_norm = baseline_cm / (baseline_cm.sum(axis=1, keepdims=True) + 1e-7)
    attack_cm_norm = attack_cm / (attack_cm.sum(axis=1, keepdims=True) + 1e-7)

    # Plot 1: Baseline
    im1 = axes[0, 0].imshow(baseline_cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0, 0].set_title('Baseline Confusion Matrix (Normalized)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot 2: Attack
    im2 = axes[0, 1].imshow(attack_cm_norm, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title(f'After Attack ({bits_flipped} bits)', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot 3: Difference matrix (Attack - Baseline)
    diff_cm = attack_cm_norm - baseline_cm_norm
    im3 = axes[1, 0].imshow(diff_cm, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('Attack-Induced Confusion (Delta)', fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0])

    # Plot 4: Per-class accuracy drop
    acc_drop_per_class = np.diag(baseline_cm_norm) - np.diag(attack_cm_norm)
    axes[1, 1].bar(range(len(acc_drop_per_class)), acc_drop_per_class * 100, color='coral')
    axes[1, 1].set_title('Per-Class Accuracy Drop (%)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(NOTEBOOK_VIZ_DIR / '1_detailed_confusion_analysis.png')
    print(f"✓ Saved: 1_detailed_confusion_analysis.png")

# -----------------------------------------------------------------------------
# VIZ 2: Misidentification Pattern Analysis
# -----------------------------------------------------------------------------

def analyze_misidentification_patterns():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    attack_cm_norm = attack_cm / (attack_cm.sum(axis=1, keepdims=True) + 1e-7)

    top_confusions = []
    for true_id in range(len(attack_cm_norm)):
        predictions = attack_cm_norm[true_id].copy()
        predictions[true_id] = 0 
        if predictions.max() > 0:
            confused_id = predictions.argmax()
            top_confusions.append((true_id, confused_id, predictions[confused_id]))

    top_confusions.sort(key=lambda x: x[2], reverse=True)
    top_n = min(15, len(top_confusions))
    
    labels = [f"ID {t}→{c}" for t, c, r in top_confusions[:top_n]]
    rates = [r * 100 for t, c, r in top_confusions[:top_n]]

    ax1.barh(range(top_n), rates, color='crimson')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_title(f'Top {top_n} Identity Confusion Pairs')

    # Sub-heatmap
    confused_ids = sorted(list(set([t for t, c, r in top_confusions[:top_n]] + [c for t, c, r in top_confusions[:top_n]])))
    sub_cm = attack_cm_norm[np.ix_(confused_ids, confused_ids)]
    sns.heatmap(sub_cm, ax=ax2, cmap='YlOrRd', xticklabels=[f'ID{i}' for i in confused_ids], yticklabels=[f'ID{i}' for i in confused_ids])
    ax2.set_title('Confusion Heatmap (Affected Subset)')

    plt.tight_layout()
    plt.savefig(NOTEBOOK_VIZ_DIR / '2_misidentification_patterns.png')
    print(f"✓ Saved: 2_misidentification_patterns.png")

# -----------------------------------------------------------------------------
# VIZ 3: Attack Impact Metrics Dashboard
# -----------------------------------------------------------------------------

def plot_impact_dashboard():
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Accuracy Bar
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['Baseline', 'Attack'], [baseline_acc*100, attack_acc*100], color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Accuracy %')
    ax1.set_ylim([0, 100])

    # ICR Bar
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(['Baseline', 'Attack'], [baseline_icr*100, attack_icr*100], color=['#3498db', '#c0392b'])
    ax2.set_title('Identity Confusion Rate %')

    # Bit Efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    bit_eff = results['attack']['bit_efficiency'] * 100
    ax3.bar(['Efficiency'], [bit_eff], color='#8e44ad')
    ax3.set_title('ICR % Gain Per Bit')

    # Summary Text
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    summary_text = (
        f"BIT FLIP ATTACK SUMMARY\n"
        f"----------------------------------\n"
        f"Total Samples: {total_samples}\n"
        f"Bits Flipped: {bits_flipped}\n"
        f"Accuracy Drop: {results['improvement']['accuracy_drop']*100:.2f}%\n"
        f"ICR Increase: +{results['improvement']['icr_increase']*100:.2f}%\n"
        f"Additional Misidentifications: +{results['attack']['misidentified_count'] - results['baseline']['misidentified_count']}"
    )
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=16, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(NOTEBOOK_VIZ_DIR / '3_attack_impact_dashboard.png')
    print(f"✓ Saved: 3_attack_impact_dashboard.png")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting Visualization Generation...")
    
    plot_detailed_confusion_comparison()
    analyze_misidentification_patterns()
    plot_impact_dashboard()
    
    print("\n" + "="*80)
    print("✨ SUCCESS: All visualizations generated!")
    print(f"📁 Check folder: {NOTEBOOK_VIZ_DIR}")
    print("="*80)