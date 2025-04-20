"""
Example script for bit flipping attacks on vision-language models.

This script provides examples of how to run the bit flipping attack 
with different models and configurations.
"""

import os
import torch
from bit_flip_attack import BitFlipAttack
from models.clip_model import load_clip_model

def clip_attack_example():
    """
    Example of bit flipping attack on CLIP model with CIFAR-10.
    """
    print("=" * 80)
    print("Example: CLIP Model with CIFAR-10")
    print("=" * 80)
    
    # Load model and dataset
    model, dataset = load_clip_model(
        dataset_name="cifar10",
        batch_size=32,
        model_name="ViT-B-32",
        pretrained="openai"
    )
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Configure attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    attack = BitFlipAttack(
        model=model,
        dataset=dataset,
        target_asr=0.7,  # Lower target for faster example
        max_bit_flips=5,  # Limit number of bits for faster example
        accuracy_threshold=0.05,
        device=device
    )
    
    # Run attack targeting class 0 (airplane)
    results = attack.perform_attack(
        target_class=0,
        num_candidates=100  # Fewer candidates for faster example
    )
    
    # Save results
    attack.save_results(results, output_dir="results")
    
    print(f"Attack completed. Results saved to results/")

def compare_target_classes():
    """
    Compare bit flipping attacks targeting different classes.
    """
    print("=" * 80)
    print("Example: Comparing different target classes")
    print("=" * 80)
    
    # Load model and dataset
    model, dataset = load_clip_model(
        dataset_name="cifar10",
        batch_size=32,
        model_name="ViT-B-32",
        pretrained="openai"
    )
    
    # Create results directory for comparison
    comparison_dir = "results_class_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Configure attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Target classes to compare (airplane, automobile, bird)
    target_classes = [0, 1, 2]
    
    for target_class in target_classes:
        print(f"\nRunning attack targeting class {target_class}")
        
        attack = BitFlipAttack(
            model=model,
            dataset=dataset,
            target_asr=0.7,
            max_bit_flips=5,
            accuracy_threshold=0.05,
            device=device
        )
        
        results = attack.perform_attack(
            target_class=target_class,
            num_candidates=100
        )
        
        # Save results in class-specific directory
        class_dir = os.path.join(comparison_dir, f"class_{target_class}")
        os.makedirs(class_dir, exist_ok=True)
        attack.save_results(results, output_dir=class_dir)
        
        print(f"Completed attack for class {target_class}")
    
    print(f"Class comparison completed. Results saved to {comparison_dir}/")

def vary_bit_flip_limits():
    """
    Compare bit flipping attacks with different maximum bit flip limits.
    """
    print("=" * 80)
    print("Example: Varying maximum bit flip limits")
    print("=" * 80)
    
    # Load model and dataset
    model, dataset = load_clip_model(
        dataset_name="cifar10",
        batch_size=32,
        model_name="ViT-B-32",
        pretrained="openai"
    )
    
    # Create results directory for comparison
    comparison_dir = "results_bit_limit_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Configure attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Different bit flip limits to compare
    bit_flip_limits = [3, 5, 10]
    
    for bit_limit in bit_flip_limits:
        print(f"\nRunning attack with maximum {bit_limit} bit flips")
        
        attack = BitFlipAttack(
            model=model,
            dataset=dataset,
            target_asr=0.7,
            max_bit_flips=bit_limit,
            accuracy_threshold=0.05,
            device=device
        )
        
        results = attack.perform_attack(
            target_class=0,
            num_candidates=100
        )
        
        # Save results in bit-limit-specific directory
        limit_dir = os.path.join(comparison_dir, f"limit_{bit_limit}")
        os.makedirs(limit_dir, exist_ok=True)
        attack.save_results(results, output_dir=limit_dir)
        
        print(f"Completed attack with {bit_limit} bit flip limit")
    
    print(f"Bit flip limit comparison completed. Results saved to {comparison_dir}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run bit flipping attack examples")
    parser.add_argument("--example", type=str, default="basic",
                      choices=["basic", "classes", "bit_limits", "all"],
                      help="Example to run")
    
    args = parser.parse_args()
    
    if args.example == "basic" or args.example == "all":
        clip_attack_example()
    
    if args.example == "classes" or args.example == "all":
        compare_target_classes()
    
    if args.example == "bit_limits" or args.example == "all":
        vary_bit_flip_limits() 