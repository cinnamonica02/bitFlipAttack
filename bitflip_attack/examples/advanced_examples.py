"""
Advanced examples for bit flipping attacks on state-of-the-art models.

This script demonstrates using the bit flipping attack on:
1. Large Language Models (LLMs) like LLaMA, Phi
2. Vision-Language Models (VLMs) like LLaVA
3. Quantized models (W8, W4, BitNet)

The examples show different sensitivity analysis methods and attack strategies.
"""

import os
import torch
from bit_flip_attack import BitFlipAttack

# Model imports
from models.clip_model import load_clip_model
from models.llm_model import load_llm_model
from models.vlm_model import load_vlm_model
from models.quantized_model import load_quantized_model

def llm_attack_example():
    """
    Example of bit flipping attack on a Large Language Model.
    """
    print("=" * 80)
    print("Example: Large Language Model (LLaMA) with MMLU")
    print("=" * 80)
    
    # Load model and dataset 
    # Note: This will download the model if not cached
    model, dataset = load_llm_model(
        model_name="meta-llama/Meta-Llama-3-8B",  # or "microsoft/phi-2" for a smaller model
        task="mmlu:astronomy",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if model is None or dataset is None:
        print("Failed to load LLM model or dataset")
        return
    
    # Create results directory
    os.makedirs("results_llm", exist_ok=True)
    
    # Configure attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    attack = BitFlipAttack(
        model=model,
        dataset=dataset,
        target_asr=0.9,
        max_bit_flips=5,
        accuracy_threshold=0.1,
        device=device,
        layer_sensitivity=True,
        hybrid_sensitivity=True,
        alpha=0.5  # Balance between gradient and magnitude sensitivity
    )
    
    # Run attack targeting improved performance on the benchmark task
    results = attack.perform_attack(
        target_class=None,  # Untargeted for LLMs
        num_candidates=200
    )
    
    # Save results
    attack.save_results(results, output_dir="results_llm")
    
    print(f"Attack completed. Results saved to results_llm/")

def vlm_attack_example():
    """
    Example of bit flipping attack on a Vision-Language Model.
    """
    print("=" * 80)
    print("Example: Vision-Language Model (LLaVA) with VQAv2")
    print("=" * 80)
    
    # Load model and dataset
    model, dataset = load_vlm_model(
        model_name="llava-hf/llava-1.5-7b-hf",
        dataset_name="vqav2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    if model is None or dataset is None:
        print("Failed to load VLM model or dataset")
        return
    
    # Create results directory
    os.makedirs("results_vlm", exist_ok=True)
    
    # Configure attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    attack = BitFlipAttack(
        model=model,
        dataset=dataset,
        target_asr=0.7,
        max_bit_flips=15,  # LLaVA paper shows it needs ~15 bits flipped
        accuracy_threshold=0.1,
        device=device,
        layer_sensitivity=True,
        hybrid_sensitivity=True
    )
    
    # Run attack
    results = attack.perform_attack(
        target_class=None,  # Untargeted for VLMs
        num_candidates=150
    )
    
    # Save results
    attack.save_results(results, output_dir="results_vlm")
    
    print(f"Attack completed. Results saved to results_vlm/")

def quantized_model_attack():
    """
    Example of bit flipping attack on quantized models (W8, W4, BitNet).
    """
    print("=" * 80)
    print("Example: Quantized Models (W8, W4, BitNet)")
    print("=" * 80)
    
    # Quantization methods to test
    quantization_methods = ["w8", "w4", "w1.58"]
    
    for quant_method in quantization_methods:
        print(f"\nTesting {quant_method} quantization")
        
        # Select appropriate model based on quantization
        if quant_method == "w1.58":
            model_name = "microsoft/BitNet-b1.58-1.5B"  # Example BitNet model
        else:
            model_name = "meta-llama/Meta-Llama-3-8B"  # Example for W8/W4
        
        # Load quantized model
        model, tokenizer = load_quantized_model(
            model_name=model_name,
            quantization=quant_method,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        if model is None:
            print(f"Failed to load {quant_method} quantized model")
            continue
        
        # Create dataset for this model
        # For simplicity, we'll reuse the LLM dataset creator
        from models.llm_model import create_mmlu_dataset
        dataset = create_mmlu_dataset(tokenizer, task_name="astronomy")
        
        if dataset is None:
            print("Failed to create dataset")
            continue
        
        # Create results directory
        results_dir = f"results_quant_{quant_method}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Configure attack
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # BitNet requires many more bits to be flipped as shown in the paper
        if quant_method == "w1.58":
            max_bits = 1000  # BitNet paper shows ~45K bits needed
            candidates = 5000
        elif quant_method == "w4":
            max_bits = 30
            candidates = 300
        else:  # w8
            max_bits = 5
            candidates = 200
        
        attack = BitFlipAttack(
            model=model,
            dataset=dataset,
            target_asr=0.7,
            max_bit_flips=max_bits,
            accuracy_threshold=0.1,
            device=device,
            layer_sensitivity=True,
            hybrid_sensitivity=True
        )
        
        # Run attack
        results = attack.perform_attack(
            target_class=None,
            num_candidates=candidates
        )
        
        # Save results
        attack.save_results(results, output_dir=results_dir)
        
        print(f"Attack completed for {quant_method}. Results saved to {results_dir}/")

def layer_sensitivity_analysis():
    """
    Example demonstrating layer sensitivity analysis on a model.
    """
    print("=" * 80)
    print("Example: Layer Sensitivity Analysis")
    print("=" * 80)
    
    # Load CLIP model for demonstration
    model, dataset = load_clip_model(
        dataset_name="cifar10",
        batch_size=32,
        model_name="ViT-B/32"
    )
    
    if model is None or dataset is None:
        print("Failed to load model or dataset")
        return
    
    # Create results directory
    os.makedirs("results_sensitivity", exist_ok=True)
    
    # Configure attack for sensitivity analysis only
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize BitFlipAttack but only for sensitivity analysis
    attack = BitFlipAttack(
        model=model,
        dataset=dataset,
        target_asr=0.7,
        max_bit_flips=1,  # Minimal as we're just analyzing
        accuracy_threshold=0.05,
        device=device,
        layer_sensitivity=True,
        hybrid_sensitivity=True
    )
    
    # Get a batch of data for gradient computation
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    inputs, targets = next(iter(dataloader))
    
    # Run layer ranking
    sensitive_layers = attack._rank_layers_by_sensitivity(n_samples=32)
    
    # Print top 5 most sensitive layers
    print("\nTop 5 most sensitive layers:")
    for i, layer in enumerate(sensitive_layers[:5]):
        print(f"{i+1}. {layer['name']} ({layer['type']}): Loss = {layer['loss']:.4f}")
    
    # Print layer sensitivity metrics
    import pandas as pd
    
    # Convert to DataFrame for easier visualization
    sensitivity_df = pd.DataFrame(sensitive_layers)
    
    # Save to CSV
    sensitivity_df.to_csv(os.path.join("results_sensitivity", "layer_sensitivity.csv"), index=False)
    
    print(f"Layer sensitivity analysis completed. Results saved to results_sensitivity/")

def progressive_bit_flipping():
    """
    Example demonstrating progressive bit flipping strategy.
    """
    print("=" * 80)
    print("Example: Progressive Bit Flipping")
    print("=" * 80)
    
    # Load model and dataset
    model, dataset = load_clip_model(
        dataset_name="cifar10",
        batch_size=32,
        model_name="ViT-B/32"
    )
    
    if model is None or dataset is None:
        print("Failed to load model or dataset")
        return
    
    # Create results directory
    os.makedirs("results_progressive", exist_ok=True)
    
    # Configure attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    attack = BitFlipAttack(
        model=model,
        dataset=dataset,
        target_asr=0.9,
        max_bit_flips=10,
        accuracy_threshold=0.05,
        device=device,
        layer_sensitivity=True,
        hybrid_sensitivity=True
    )
    
    # Target class (plane in CIFAR-10)
    target_class = 0
    
    # We'll manually track progressive statistics
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Initialize tracking
    flip_history = []
    accuracy_history = []
    asr_history = []
    
    # Get initial performance
    initial_accuracy, initial_asr = attack._evaluate(target_class)
    
    # Add initial state
    flip_history.append(0)
    accuracy_history.append(initial_accuracy)
    asr_history.append(initial_asr)
    
    # Set max bits to flip
    max_flips = 10
    
    # Iterate incrementally
    for bits in range(1, max_flips + 1):
        # Modify max_bit_flips for this iteration
        attack.max_bit_flips = bits
        
        # Run attack
        results = attack.perform_attack(
            target_class=target_class,
            num_candidates=100
        )
        
        # Record metrics
        flip_history.append(bits)
        accuracy_history.append(results['final_accuracy'])
        asr_history.append(results['final_asr'])
        
        print(f"Bits: {bits}, Accuracy: {results['final_accuracy']:.4f}, ASR: {results['final_asr']:.4f}")
    
    # Create and save the progressive flipping plot
    plt.figure(figsize=(10, 6))
    plt.plot(flip_history, accuracy_history, 'bo-', label='Accuracy')
    plt.plot(flip_history, asr_history, 'ro-', label='Attack Success Rate')
    plt.xlabel('Number of Bit Flips')
    plt.ylabel('Rate')
    plt.title('Progressive Bit Flipping Attack')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results_progressive", "progressive_bit_flips.png"), dpi=300)
    
    # Save data to CSV
    progress_df = pd.DataFrame({
        'Bits Flipped': flip_history,
        'Accuracy': accuracy_history,
        'ASR': asr_history
    })
    
    progress_df.to_csv(os.path.join("results_progressive", "progressive_results.csv"), index=False)
    
    print(f"Progressive bit flipping completed. Results saved to results_progressive/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run advanced bit flipping attack examples")
    parser.add_argument("--example", type=str, default="clip",
                      choices=["clip", "llm", "vlm", "quantized", "sensitivity", "progressive", "all"],
                      help="Example to run")
    
    args = parser.parse_args()
    
    # Default example (CLIP) is already in the main example.py
    if args.example == "llm" or args.example == "all":
        llm_attack_example()
    
    if args.example == "vlm" or args.example == "all":
        vlm_attack_example()
    
    if args.example == "quantized" or args.example == "all":
        quantized_model_attack()
    
    if args.example == "sensitivity" or args.example == "all":
        layer_sensitivity_analysis()
    
    if args.example == "progressive" or args.example == "all":
        progressive_bit_flipping() 