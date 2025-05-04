"""
Example script for using the UmupBitFlipAttack to target quantized and unit-scaled models

This script demonstrates how to:
1. Load a pre-trained BERT model
2. Quantize it using 8-bit quantization
3. Apply our u-μP-aware bit flip attack
4. Compare results against the standard bit flip attack
"""

import os
import torch
import numpy as np
import copy  # Added for deep copying model state
import argparse
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import traceback # For printing detailed errors

# Now import the attack classes
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack

# Import bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    has_bnb = True
except ImportError:
    has_bnb = False
    print("Warning: bitsandbytes not found. 8-bit quantization will not be available.")

# Try importing the unit_scaling library
try:
    import unit_scaling as uu
    import unit_scaling.functional as U
    has_unit_scaling = True
except ImportError:
    has_unit_scaling = False
    print("Warning: unit_scaling library not found. u-μP model creation will not be available.")


class PIIDataset(Dataset):
    """Dataset for PII detection using transformers"""
    def __init__(self, csv_file, tokenizer, max_length=128):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Dataset file {csv_file} not found.")

        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create text samples and labels
        self.texts = self.data['text'].tolist()
        self.labels = self.data['contains_pii'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Return a single dictionary including labels (for Trainer compatibility)
        return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
        }

def load_or_create_dataset(tokenizer, path="data/pii_dataset.csv", num_pii=800, num_non_pii=800):
    """
    Loads the PII dataset if it exists, otherwise creates a new noisy one
    using the SyntheticPIIGenerator's generate_noisy_classification_dataset method.
    """
    if not os.path.exists(path):
        print(f"Dataset not found at {path}, creating new noisy synthetic dataset...")

        try:
            from bitflip_attack.datasets.synthetic_pii import SyntheticPIIGenerator
            import random # Keep random import if still needed by generator
        except ImportError:
            raise ImportError("Could not import SyntheticPIIGenerator or random. Please check dependencies.")

        generator = SyntheticPIIGenerator(seed=42)

        # --- Call the new generator method --- 
        combined_df = generator.generate_noisy_classification_dataset(
            num_pii=num_pii,
            num_non_pii=num_non_pii
        )
        # --- End call to new generator method ---

        output_dir = os.path.dirname(path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)

        combined_df.to_csv(path, index=False)
        print(f"Created noisy synthetic dataset with {len(combined_df)} samples and saved to {path}")

        return PIIDataset(path, tokenizer)
    else:
        print(f"Loading dataset from {path}")
        return PIIDataset(path, tokenizer)


def create_dataloaders(dataset, batch_size=16, train_fraction=0.7):
    """Create train, validation, and test dataloaders with proper splits"""
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_fraction * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_with_metrics(model, dataloader, device):
    """Evaluate model on a dataset with detailed metrics"""
    model.eval()
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with torch.no_grad():
        for batch in dataloader:
            # Extract inputs and labels from the dictionary batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # Get labels from the dictionary

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # Calculate detailed metrics
            for pred, label in zip(predictions, labels):
                if pred == 1 and label == 1:
                    true_positives += 1
                elif pred == 1 and label == 0:
                    false_positives += 1
                elif pred == 0 and label == 0:
                    true_negatives += 1
                elif pred == 0 and label == 1:
                    false_negatives += 1

    accuracy = correct / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"True Positives: {true_positives}, False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}, False Negatives: {false_negatives}")

    return accuracy, precision, recall, f1


# --- MODIFIED custom_forward_fn ---
def custom_forward_fn(model, batch):
    """
    Custom forward function for BERT models used by the attack classes.
    Handles both dictionary inputs and tuple inputs (inputs_dict, targets).
    """
    input_dict = None
    # Check if batch is the tuple format (inputs_dict, targets)
    if isinstance(batch, tuple) and len(batch) == 2:
        # Assume the first element is the dictionary with input tensors
        input_dict = batch[0]
    # Check if batch is already the input dictionary
    elif isinstance(batch, dict):
        input_dict = batch
    else:
        # If it's neither, we can't handle it
        raise TypeError(f"Unsupported batch type in custom_forward_fn: {type(batch)}. Expected dict or tuple(dict, tensor).")

    # Now, input_dict should be the dictionary containing tensors
    if isinstance(input_dict, dict) and 'input_ids' in input_dict and 'attention_mask' in input_dict:
        input_ids = input_dict['input_ids'].to(model.device)
        attention_mask = input_dict['attention_mask'].to(model.device)
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    else:
        # If input_dict is not the expected format after potential extraction
        raise ValueError(f"Could not extract valid input_dict with input_ids/attention_mask in custom_forward_fn. Received input_dict type: {type(input_dict)}")
# --- END MODIFIED custom_forward_fn ---


def quantize_model(model, quantization_type="8bit"):
    """
    Quantize model using bitsandbytes or unit-scaling simulation.

    Args:
        model: The model to quantize (should be on CPU before calling for BNB).
        quantization_type: Quantization type ('8bit', '4bit', or 'none').

    Returns:
        Quantized model (on the original device).
    """
    if quantization_type == "none":
        return model

    original_device = next(model.parameters()).device
    print(f"Original model device: {original_device}")
    print(f"Applying {quantization_type} quantization...")

    # Preferred: unit-scaling if available
    if has_unit_scaling:
        print("Attempting unit-scaling based quantization...")
        try:
            from unit_scaling.transforms import unit_scale, simulate_fp8
            # Unit scale requires model on CPU
            model_cpu = model.cpu()
            unit_scaled_model = unit_scale(model_cpu)

            if quantization_type == "8bit":
                print("Simulating 8-bit format (E4M3)...")
                quantized_model = simulate_fp8(unit_scaled_model, exponent_bits=4, mantissa_bits=3)
            elif quantization_type == "4bit":
                print("Simulating 4-bit format (E3M0)...")
                quantized_model = simulate_fp8(unit_scaled_model, exponent_bits=3, mantissa_bits=0)
            else:
                 raise ValueError("Unsupported quantization type for unit-scaling simulation")

            print("Unit-scaling quantization successful.")
            return quantized_model.to(original_device) # Move back to original device
        except Exception as e:
            print(f"Error applying unit-scaling quantization: {e}. Falling back...")
            # Ensure model is back on original device if fallback needed
            model = model.to(original_device)

        # Fallback: bitsandbytes if available
    if has_bnb:
        print("Attempting bitsandbytes based quantization...")
        # Move model to CPU for safer conversion for BNB manual replacement
        model_cpu = model.cpu() # Work on a CPU copy
        try:
            if quantization_type == "8bit":
                from bitsandbytes.nn import Linear8bitLt
                print("Replacing Linear layers with Linear8bitLt...")
                for name, module in model_cpu.named_modules(): # Iterate over CPU copy
                    if isinstance(module, torch.nn.Linear):
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = model_cpu.get_submodule(parent_name) if parent_name else model_cpu # Get parent from CPU copy

                        new_layer = Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False # Important for stability
                        )
                        # Copy weights and biases FROM the original module TO the new layer
                        new_layer.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_layer.bias.data.copy_(module.bias.data)

                        setattr(parent, child_name, new_layer) # Set attribute on CPU copy
                print("8-bit quantization using bitsandbytes successful.")

            elif quantization_type == "4bit":
                 from transformers import BitsAndBytesConfig
                 from bitsandbytes.nn import Linear4bit
                 print("Replacing Linear layers with Linear4bit...")

                 quantization_config = BitsAndBytesConfig(
                     load_in_4bit=True,
                     bnb_4bit_compute_dtype=torch.float16,
                     bnb_4bit_use_double_quant=True,
                     bnb_4bit_quant_type="nf4"
                 )

                 for name, module in model_cpu.named_modules(): # Iterate over CPU copy
                     if isinstance(module, torch.nn.Linear):
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = model_cpu.get_submodule(parent_name) if parent_name else model_cpu # Get parent from CPU copy

                        new_layer = Linear4bit(
                             module.in_features,
                             module.out_features,
                             bias=module.bias is not None,
                             compute_dtype=quantization_config.bnb_4bit_compute_dtype,
                             quant_type=quantization_config.bnb_4bit_quant_type,
                             use_double_quant=quantization_config.bnb_4bit_use_double_quant,
                        )
                        # Copy weights and biases FROM the original module TO the new layer
                        new_layer.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_layer.bias.data.copy_(module.bias.data)

                        setattr(parent, child_name, new_layer) # Set attribute on CPU copy
                 print("4-bit quantization using bitsandbytes successful.")

            else:
                 raise ValueError(f"Unsupported quantization type for bitsandbytes: {quantization_type}")

            # --- Critical Fix: Move the modified CPU model back to the original device ---
            quantized_model = model_cpu.to(original_device)
            print(f"Model moved back to {original_device}")
            return quantized_model
            # --- End Critical Fix ---

        except Exception as e:
            print(f"Error applying bitsandbytes quantization: {e}")
            print("Falling back to PyTorch fake quantization (if possible).")
            # Ensure model is back on original device if fallback needed
            model = model.to(original_device) # Return original model on correct device
            # Fallthrough to PyTorch fake quantization (which should handle device correctly)


    # Last resort: PyTorch fake quantization (mainly for testing, not true reduced precision)
    print("Attempting PyTorch fake quantization...")
    try:
        return _apply_pytorch_fake_quantization(model)
    except Exception as e:
        print(f"Error applying PyTorch fake quantization: {e}")
        print("WARNING: Quantization failed entirely. Returning original model.")
        return model.to(original_device) # Ensure it's on the right device


def _apply_pytorch_fake_quantization(model):
    """Helper function to apply PyTorch's fake quantization"""
    original_device = next(model.parameters()).device
    print(f"Applying PyTorch fake quantization on device: {original_device}")
    # Move model to CPU for quantization prep if needed by backend
    model = model.cpu()

    from torch.quantization import FakeQuantize, default_observer, default_qconfig

    # Define fake quantization for FP32 --> INT8
    qconfig = default_qconfig
    model.qconfig = qconfig

    # Prepare and convert model
    # Note: Proper calibration (`torch.quantization.calibrate`) is usually needed
    # but skipped here for simplicity as it's a fallback.
    model = torch.quantization.prepare_qat(model, inplace=True) # Use prepare_qat for robustness
    model = torch.quantization.convert(model, inplace=True)

    # Move model back to the original device
    model = model.to(original_device)

    print(f"Model converted using PyTorch's fake quantization and moved to {original_device}")
    return model


def convert_to_umup(model):
    """
    Convert a standard model to u-μP using unit-scaling library.
    """
    if not has_unit_scaling:
        raise ImportError("unit_scaling library is required to convert models to u-μP")

    print("Converting model to u-μP using unit_scale transform...")
    try:
        # unit_scale usually expects model on CPU
        original_device = next(model.parameters()).device
        model = model.cpu()
        umup_model = unit_scale(model)
        print("u-μP conversion successful.")
        return umup_model.to(original_device) # Move back to original device
    except Exception as e:
        print(f"Error during u-μP conversion: {e}")
        print("WARNING: u-μP conversion failed. Returning original model.")
        return model.to(original_device)


def run_attack_comparison(model_name="bert-base-uncased", quantization_type="8bit",
                          use_umup=False, batch_size=16, max_bit_flips=5):
    """
    Run and compare different bit flip attacks on a model

    Args:
        model_name: Name of the pre-trained model to use
        quantization_type: Type of quantization to apply ('none', '8bit', '4bit', 'fake_quant')
        use_umup: Whether to convert model to u-μP (requires unit-scaling library)
        batch_size: Batch size for training and evaluation
        max_bit_flips: Maximum number of bits to flip

    Returns:
        Results dictionary
    """
    print(f"\n==== Starting Attack Comparison ====")
    print(f"Model: {model_name}")
    print(f"Quantization: {quantization_type}")
    print(f"Use u-μP: {use_umup}")
    print(f"Max Bit Flips: {max_bit_flips}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Use a smaller model if base specified
    initial_model_name = model_name # Keep original name if needed later
    if "bert-base" in model_name:
        print("Using smaller model: prajjwal1/bert-tiny for faster experimentation")
        model_name = "prajjwal1/bert-tiny"

    # Load base model
    print(f"Loading base model: {model_name}")
    # Load model initially on CPU for potential modifications (like quantization)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device) # Move to target device before training
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters on device {device}")

    # Load or create dataset
    dataset = load_or_create_dataset(tokenizer, num_pii=5000, num_non_pii=5000)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=batch_size,
        train_fraction=0.5 # Use less training data
    )

    # --- Setup Trainer ---
    print("Setting up Hugging Face Trainer...")
    # Adjust TrainingArguments for compatibility and 1 epoch training
    training_args = TrainingArguments(
        output_dir="./results",          # output directory
        num_train_epochs=1,              # Train for 1 epoch
        per_device_train_batch_size=batch_size,  # Use arg batch size
        per_device_eval_batch_size=batch_size*2, # Can be larger for eval
        warmup_steps=50,                 # Reduced warmup steps for 1 epoch
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,                # Log frequently
        logging_strategy="epoch",        # Log each epoch (should trigger eval with Trainer)
        # evaluation_strategy="epoch",   # Removed for compatibility/simplicity
        # save_strategy="epoch",         # Removed for compatibility/simplicity
        # load_best_model_at_end=True,   # Removed as we only have 1 epoch
        # metric_for_best_model="accuracy",# Removed (related to load_best)
        save_total_limit=1,              # Keep only the final checkpoint
        seed=42,                         # For reproducibility
        data_seed=42                     # For reproducibility
    )

    trainer = Trainer(
        model=model, # Pass the model already on the correct device
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset
        # Default data collator should work now with PIIDataset returning dicts
    )

    # --- Start Fine-Tuning ---
    print("Starting fine-tuning using Hugging Face Trainer...")
    trainer.train()

    # --- Save the final model state ---
    print("Saving the final model after training...")
    # Trainer saves automatically to output_dir/checkpoint-XXX, find the last one or use default save
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_model_path) # Saves the final model state
    tokenizer.save_pretrained(final_model_path) # Save tokenizer too
    print(f"Final model saved to {final_model_path}")

    # --- Load the final model state after training for evaluation ---
    print(f"\nLoading final model from {final_model_path} for evaluation and attack.")
    # Load fresh instance to ensure clean state before loading weights
    model = BertForSequenceClassification.from_pretrained(final_model_path, num_labels=2)
    model.to(device)

    # Evaluate the fine-tuned model on the test set
    print("\nEvaluating fine-tuned model on test set...")
    original_accuracy, precision, recall, f1 = evaluate_with_metrics(model, test_loader, device)
    print(f"Fine-tuned model - Accuracy: {original_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # --- Determine the starting model/state for the attacks ---
    # Start with the fine-tuned model state
    attack_start_model = model
    # Store initial state dict for potentially resetting between attacks
    attack_start_state_dict = copy.deepcopy(model.state_dict())
    quantized_accuracy, q_precision, q_recall, q_f1 = original_accuracy, precision, recall, f1 # Default metrics if no quantization

    # --- Apply quantization if requested ---
    if quantization_type != "none":
        print(f"\nApplying {quantization_type} quantization...")
        # Create a fresh model instance to quantize, starting from fine-tuned state
        # Important: quantize_model expects model on CPU for BNB manual layer replacement
        model_to_quantize = BertForSequenceClassification.from_pretrained(final_model_path, num_labels=2).cpu()

        try:
            quantized_model = quantize_model(model_to_quantize, quantization_type) # Returns model on original device
            print(f"Quantization successful. Model is on {next(quantized_model.parameters()).device}")
            # Evaluate quantized model
            quantized_accuracy, q_precision, q_recall, q_f1 = evaluate_with_metrics(quantized_model, test_loader, device)
            print(f"Quantized model - Accuracy: {quantized_accuracy:.4f}, Precision: {q_precision:.4f}, Recall: {q_recall:.4f}, F1: {q_f1:.4f}")

            # Set the quantized model as the starting point for attacks
            attack_start_model = quantized_model # This is now our baseline for attacks
            # Update the state dict to the quantized one
            attack_start_state_dict = copy.deepcopy(quantized_model.state_dict())

        except Exception as q_err:
            print(f"ERROR during quantization: {q_err}. Proceeding with non-quantized model for attacks.")
            traceback.print_exc()
            # Metrics remain the original ones
            quantized_accuracy, q_precision, q_recall, q_f1 = original_accuracy, precision, recall, f1
            # Ensure we use the original fine-tuned model
            attack_start_model = model # Revert back to the original fine-tuned model object
            attack_start_state_dict = copy.deepcopy(model.state_dict()) # Ensure state dict matches

    # --- Convert to u-μP if requested (Applied AFTER potential quantization) ---
    if use_umup:
        print("\nAttempting u-μP conversion...")
        try:
            # Apply conversion to the model state we intend to attack (could be quantized or not)
            attack_start_model = convert_to_umup(attack_start_model) # Returns model on original device
            print(f"u-μP conversion applied. Model is on {next(attack_start_model.parameters()).device}")
            # Update the state dict to the u-μP converted one
            attack_start_state_dict = copy.deepcopy(attack_start_model.state_dict())
            # Re-evaluate after conversion? Optional, depends on if conversion affects accuracy significantly
            # umup_accuracy, umup_precision, umup_recall, umup_f1 = evaluate_with_metrics(attack_start_model, test_loader, device)
            # print(f"u-μP Converted Model - Accuracy: {umup_accuracy:.4f}, Precision: {umup_precision:.4f}, Recall: {umup_recall:.4f}, F1: {umup_f1:.4f}")

        except Exception as e:
            print(f"Failed to convert model to u-μP: {e}. Continuing without u-μP conversion.")
            traceback.print_exc()
            # Ensure model is still on the correct device if conversion failed mid-way
            attack_start_model.to(device)
            # State dict remains as it was before attempting conversion


    # --- Dataset Wrapping for Attack Classes ---
    # The attack classes might expect a specific dataset structure
    attack_dataset = test_loader.dataset
    print(f"Attack dataset type: {type(attack_dataset)}")
    # Use the updated AttackDataset wrapper
    wrapped_dataset = AttackDataset(attack_dataset)


    # --- Target Modules ---
    print("\nIdentifying target modules...")
    target_modules = []
    # Use the model we intend to attack (could be fine-tuned, quantized, or umup-converted)
    current_model_for_attack_setup = attack_start_model
    classifier = current_model_for_attack_setup.classifier if hasattr(current_model_for_attack_setup, 'classifier') else None
    if classifier:
        target_modules.append(classifier)
        print("Targeting classifier module")
    if hasattr(current_model_for_attack_setup, 'bert') and hasattr(current_model_for_attack_setup.bert, 'encoder') and hasattr(current_model_for_attack_setup.bert.encoder, 'layer'):
        last_layer = current_model_for_attack_setup.bert.encoder.layer[-1]
        target_modules.append(last_layer)
        print(f"Targeting last transformer layer: {last_layer.__class__.__name__}")


    # Initialize results containers
    standard_results = {"error": "Attack not run"}
    umup_results = {"error": "Attack not run or N/A"}

    # --- Create a model instance that will be modified by attacks ---
    # Start by deep copying the initial state (could be fine-tuned, quantized, u-μP)
    attack_model_for_attacks = copy.deepcopy(attack_start_model)
    print(f"Created deep copy of model for attacks on device: {next(attack_model_for_attacks.parameters()).device}")

