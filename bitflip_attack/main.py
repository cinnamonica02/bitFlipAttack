import argparse
import torch
import os
import sys
from bit_flip_attack import BitFlipAttack

# Ensure models directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

def main():
    parser = argparse.ArgumentParser(description="Bit Flipping Attack on Vision-Language Models")
    parser.add_argument("--model", type=str, default="clip", 
                      choices=["clip", "llava", "cogvlm", "minigpt4"],
                      help="Target model architecture")
    parser.add_argument("--model_variant", type=str, default="ViT-B-32",
                      help="Specific model variant/size")
    parser.add_argument("--dataset", type=str, default="cifar10",
                      choices=["cifar10", "imagenet", "mscoco"],
                      help="Dataset for evaluation")
    parser.add_argument("--target_asr", type=float, default=0.9,
                      help="Target attack success rate")
    parser.add_argument("--max_bit_flips", type=int, default=100,
                      help="Maximum number of bits to flip")
    parser.add_argument("--accuracy_threshold", type=float, default=0.05,
                      help="Maximum acceptable accuracy drop")
    parser.add_argument("--target_class", type=int, default=0,
                      help="Target class for attack")
    parser.add_argument("--num_candidates", type=int, default=1000,
                      help="Number of bit candidates to evaluate")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for dataloaders")
    parser.add_argument("--results_dir", type=str, default="results",
                      help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Import model and dataset based on arguments
    print(f"Bit flipping attack using {args.model} on {args.dataset}")
    
    # Load model and dataset
    if args.model == "clip":
        try:
            from models.clip_model import load_clip_model
            model, dataset = load_clip_model(
                dataset_name=args.dataset,
                batch_size=args.batch_size,
                model_name=args.model_variant,
                pretrained="openai"
            )
        except ImportError:
            print("Error: Could not import clip_model module. Make sure the models directory exists.")
            sys.exit(1)
    elif args.model == "llava":
        try:
            # This would be implemented in a separate module
            print("LLaVA model support is not yet implemented.")
            sys.exit(1)
        except ImportError:
            print("Error: Could not import llava_model module.")
            sys.exit(1)
    elif args.model == "cogvlm":
        try:
            # This would be implemented in a separate module
            print("CogVLM model support is not yet implemented.")
            sys.exit(1)
        except ImportError:
            print("Error: Could not import cogvlm_model module.")
            sys.exit(1)
    elif args.model == "minigpt4":
        try:
            # This would be implemented in a separate module
            print("MiniGPT-4 model support is not yet implemented.")
            sys.exit(1)
        except ImportError:
            print("Error: Could not import minigpt4_model module.")
            sys.exit(1)
    else:
        print(f"Unsupported model: {args.model}")
        sys.exit(1)
    
    # Check if model and dataset are loaded successfully
    if model is None or dataset is None:
        print("Error: Failed to load model or dataset.")
        sys.exit(1)
    
    # Initialize and perform attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    attack = BitFlipAttack(
        model=model,
        dataset=dataset,
        target_asr=args.target_asr,
        max_bit_flips=args.max_bit_flips,
        accuracy_threshold=args.accuracy_threshold,
        device=device
    )
    
    results = attack.perform_attack(
        target_class=args.target_class,
        num_candidates=args.num_candidates
    )
    
    # Save results
    attack.save_results(results, output_dir=args.results_dir)
    
    print(f"Attack completed. Results saved to {args.results_dir}/")

if __name__ == "__main__":
    main() 