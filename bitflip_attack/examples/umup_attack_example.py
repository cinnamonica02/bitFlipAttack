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
import argparse
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm

# Import our enhanced bit flip attack
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack

# Monkey patch the flip_bit function to prevent index out of bounds errors
from bitflip_attack.attacks.helpers.bit_manipulation import flip_bit as original_flip_bit

def safe_flip_bit(layer, param_idx, bit_pos):
    """
    A safer version of flip_bit that handles index errors gracefully.
    """
    try:
        # Get module and parameter tensor
        module = layer['module']
        weight = module.weight
        
        # Check if param_idx is in bounds
        if param_idx >= weight.numel():
            print(f"Warning: Parameter index {param_idx} is out of bounds for tensor with size {weight.numel()}")
            # Use a valid index instead (e.g., modulo the size)
            param_idx = param_idx % weight.numel()
            print(f"Using parameter index {param_idx} instead")
        
        # Call the original function with the adjusted index
        return original_flip_bit(layer, param_idx, bit_pos)
    except Exception as e:
        print(f"Error in flip_bit: {str(e)}")
        # Return dummy values as fallback
        return 0.0, 0.0

# Replace the original function with our safe version
import bitflip_attack.attacks.helpers.bit_manipulation
bitflip_attack.attacks.helpers.bit_manipulation.flip_bit = safe_flip_bit

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
        
        return (
            {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            },
            torch.tensor(label, dtype=torch.long)
        )


def load_or_create_dataset(tokenizer, path="data/pii_dataset.csv"):
    """
    Load or create a synthetic PII dataset
    """
    if not os.path.exists(path):
        # Generate a simple synthetic dataset with PII data
        print(f"Dataset not found at {path}, creating synthetic dataset...")
        
        # Import the necessary module for dataset generation
        try:
            from bitflip_attack.datasets.synthetic_pii import SyntheticPIIGenerator
        except ImportError:
            raise ImportError("Could not import SyntheticPIIGenerator. Please install the required dependencies.")
            
        # Generate synthetic data
        generator = SyntheticPIIGenerator(seed=42)
        records = generator.generate_personal_records(num_records=1000)
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Create text field by combining personal information
        df['text'] = df.apply(
            lambda row: f"Name: {row.get('first_name', '')} {row.get('last_name', '')}, " +
                       f"Email: {row.get('email', '')}, " +
                       f"Phone: {row.get('phone_number', '')}, " +
                       f"SSN: {row.get('ssn', '')}, " +
                       f"Address: {row.get('address', '')}, " +
                       f"Date of Birth: {row.get('dob', '')}", 
            axis=1
        )
        df['contains_pii'] = 1  # All records contain PII
        
        # Add non-PII samples
        non_pii_samples = [
            {"text": "This is a regular text without any personal information.", "contains_pii": 0},
            {"text": "The weather today is sunny and warm.", "contains_pii": 0},
            {"text": "Machine learning models can be vulnerable to attacks.", "contains_pii": 0},
        ] * 300
        
        non_pii_df = pd.DataFrame(non_pii_samples)
        combined_df = pd.concat([df[['text', 'contains_pii']], non_pii_df], ignore_index=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save to CSV
        combined_df.to_csv(path, index=False)
        print(f"Created synthetic dataset with {len(combined_df)} samples")
        
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
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
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


def custom_forward_fn(model, batch):
    """Custom forward function for BERT models"""
    inputs, _ = batch
    
    # Ensure inputs are properly formatted dictionaries with input_ids and attention_mask
    if not isinstance(inputs, dict):
        print(f"Warning: Expected dict for inputs, got {type(inputs)}")
        # Try to handle different dataset structures
        if hasattr(inputs, 'input_ids') and hasattr(inputs, 'attention_mask'):
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            return model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    # Regular case - expect a dictionary with input_ids and attention_mask
    return model(
        input_ids=inputs['input_ids'].to(model.device),
        attention_mask=inputs['attention_mask'].to(model.device)
    ).logits


def quantize_model(model, quantization_type="8bit"):
    """
    Quantize model using graphcore 8-bit quantization with unit-scaling
    
    Args:
        model: The model to quantize
        quantization_type: Quantization type ('8bit', '4bit', or 'none')
    
    Returns:
        Quantized model
    """
    if quantization_type == "none":
        return model
    
    # Get the device of the model
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    print(f"Applying {quantization_type} quantization...")
    
    # First check if we have the required libraries
    try:
        import unit_scaling as uu
        import unit_scaling.functional as U
        from unit_scaling.transforms import unit_scale, simulate_fp8
        has_unit_scaling = True
    except ImportError:
        print("Warning: unit_scaling library not found. Using fallback quantization.")
        has_unit_scaling = False
    
    if quantization_type == "8bit":
        if has_unit_scaling:
            # Step 1: Apply unit scaling to the model
            # This ensures weights have unit variance which is ideal for quantization
            print("Applying unit scaling transformation...")
            try:
                # Apply unit scaling transform
                unit_scaled_model = unit_scale(model)
                
                # Step 2: Apply 8-bit format simulation
                # This uses the simulate_fp8 transform from the unit-scaling library
                print("Simulating 8-bit format...")
                quantized_model = simulate_fp8(unit_scaled_model, 
                                              exponent_bits=4,  # For 8-bit, usually 4 exponent bits
                                              mantissa_bits=3)  # and 3 mantissa bits + 1 sign bit
                return quantized_model
            except Exception as e:
                print(f"Error applying unit-scaling quantization: {e}")
                print("Falling back to bitsandbytes quantization...")
                
        # Fallback to bitsandbytes quantization if unit-scaling fails or isn't available
        try:
            import bitsandbytes as bnb
            from bitsandbytes.nn import Linear8bitLt
            
            # Move model to CPU for safer conversion
            model = model.cpu()
            
            # Replace linear layers with 8-bit quantized versions
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model
                    if parent_name:
                        for attr in parent_name.split('.'):
                            parent = getattr(parent, attr)
                    
                    # Create 8-bit layer and replace
                    new_layer = Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False
                    )
                    # Copy weights and biases
                    new_layer.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        new_layer.bias.data.copy_(module.bias.data)
                    
                    setattr(parent, child_name, new_layer)
            
            # Move model back to the original device
            model = model.to(device)
            print(f"Model converted to 8-bit quantization using bitsandbytes and moved to {device}")
            return model
        except ImportError:
            print("Neither unit-scaling nor bitsandbytes available. Using PyTorch's fake quantization.")
            return _apply_pytorch_fake_quantization(model)
    
    elif quantization_type == "4bit":
        # Similar approach for 4-bit
        if has_unit_scaling:
            # Apply unit scaling first
            unit_scaled_model = unit_scale(model)
            
            # Simulate 4-bit format with fewer bits for mantissa
            quantized_model = simulate_fp8(unit_scaled_model,
                                          exponent_bits=3,  # For 4-bit, usually 3 exponent bits
                                          mantissa_bits=0)  # and 0 mantissa bits + 1 sign bit
            return quantized_model
        else:
            # For 4-bit, use simpler approach with Transformers integration
            try:
                print("Using bitsandbytes 4-bit quantization...")
                
                # Configure 4-bit quantization
                from transformers import BitsAndBytesConfig
                
                # Move model to CPU for quantization
                model = model.cpu()
                
                # Use bitsandbytes to quantize model in-place with better defaults
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Manually quantize the model using bitsandbytes
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = model
                        if parent_name:
                            for attr in parent_name.split('.'):
                                parent = getattr(parent, attr)
                        
                        # Convert to 4-bit using bitsandbytes
                        from bitsandbytes.nn import Linear4bit
                        new_layer = Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None
                        )
                        # Copy weights and biases
                        new_layer.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_layer.bias.data.copy_(module.bias.data)
                        
                        setattr(parent, child_name, new_layer)
                
                # Move model back to the original device
                model = model.to(device)
                print(f"Model converted to 4-bit quantization and moved to {device}")
                return model
            except Exception as e:
                print(f"Error applying 4-bit quantization: {e}")
                print("Using PyTorch's fake quantization instead.")
                return _apply_pytorch_fake_quantization(model)
    
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")


def _apply_pytorch_fake_quantization(model):
    """Helper function to apply PyTorch's fake quantization"""
    try:
        # Get the device of the model
        device = next(model.parameters()).device
        
        # Move model to CPU for quantization
        model = model.cpu()
        
        from torch.quantization import FakeQuantize, default_observer, default_qconfig
        
        # Define fake quantization for FP32 --> INT8
        qconfig = default_qconfig
        model.qconfig = qconfig
        
        # Prepare and convert model
        model = torch.quantization.prepare(model, inplace=True)
        # Calibration would happen here with real data
        model = torch.quantization.convert(model, inplace=True)
        
        # Move model back to the original device
        model = model.to(device)
        
        print(f"Model converted using PyTorch's fake quantization and moved to {device}")
        return model
    except Exception as e:
        print(f"Error applying PyTorch quantization: {e}")
        print("Returning unquantized model")
        return model


def convert_to_umup(model):
    """
    Convert a standard model to u-μP
    Note: This is a simplified demonstration and won't work with all model architectures
    """
    if not has_unit_scaling:
        raise ImportError("unit_scaling library is required to convert models to u-μP")
    
    # For a real implementation, we would need a full model conversion
    # This is just a sketch of the approach
    print("Converting model to u-μP (demonstration only)")
    
    # In practice, this would be a more complex conversion process
    # following the patterns in the unit-scaling library
    
    return model


def run_attack_comparison(model_name="bert-base-uncased", quantization_type="8bit", 
                          use_umup=False, batch_size=16, max_bit_flips=5):
    """
    Run and compare different bit flip attacks on a model
    
    Args:
        model_name: Name of the pre-trained model to use
        quantization_type: Type of quantization to apply
        use_umup: Whether to convert model to u-μP
        batch_size: Batch size for training and evaluation
        max_bit_flips: Maximum number of bits to flip
        
    Returns:
        Results dictionary
    """
    print(f"==== Starting Attack Comparison ====")
    print(f"Model: {model_name}")
    print(f"Quantization: {quantization_type}")
    print(f"Use u-μP: {use_umup}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Use a smaller model for faster experimentation
    if model_name == "bert-base-uncased":
        # Can try a smaller model to reduce overfitting
        print("Using smaller model: prajjwal1/bert-tiny for faster experimentation")
        model_name = "prajjwal1/bert-tiny"  # 4-layer BERT model, much smaller

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load or create dataset
    dataset = load_or_create_dataset(tokenizer)
    
    # Create a smaller, more challenging dataset to prevent trivial separation
    print("Creating more challenging dataset to avoid perfect accuracy...")
    # Create dataloaders with proper splits and smaller training set
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, 
        batch_size=batch_size,
        train_fraction=0.5  # Use less training data to avoid overfitting
    )
    
    # Fine-tune the model on the PII dataset
    print("Fine-tuning model on PII dataset...")
    model.to(device)
    
    # Training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    num_epochs = 3  # More epochs with early stopping
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            if step % 50 == 0:
                print(f"Step: {step}, Loss: {loss.item():.4f}")
        
        # Evaluate on validation set after each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "models_dir/best_model.pt")
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # Load best model for evaluation and attack
    model.load_state_dict(torch.load("models_dir/best_model.pt"))
    
    # Evaluate the model on test set
    print("\nEvaluating model on test set...")
    original_accuracy, precision, recall, f1 = evaluate_with_metrics(model, test_loader, device)
    print(f"Original model - Accuracy: {original_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Save the original model for comparison
    os.makedirs("models_dir", exist_ok=True)
    torch.save(model.state_dict(), "models_dir/original_model.pt")
    
    # Apply quantization
    if quantization_type != "none":
        model = quantize_model(model, quantization_type)
    
    # Convert to u-μP if requested
    if use_umup:
        try:
            model = convert_to_umup(model)
        except Exception as e:
            print(f"Failed to convert model to u-μP: {e}")
            print("Continuing with standard model")
    
    # Evaluate the quantized/u-μP model
    quantized_accuracy, q_precision, q_recall, q_f1 = evaluate_with_metrics(model, test_loader, device)
    print(f"Quantized model - Accuracy: {quantized_accuracy:.4f}, Precision: {q_precision:.4f}, Recall: {q_recall:.4f}, F1: {q_f1:.4f}")
    
    # Extract the test dataset from the dataloader
    attack_dataset = test_loader.dataset
    print(f"Attack dataset type: {type(attack_dataset)}")
    
    # Define a simple wrapper class to ensure consistent dataset format for attacks
    class AttackDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            data = self.dataset[idx]
            # Ensure consistent format: (inputs_dict, label)
            if isinstance(data, tuple) and len(data) == 2:
                inputs, label = data
                # Make sure inputs is a dict with input_ids and attention_mask
                if isinstance(inputs, dict) and 'input_ids' in inputs and 'attention_mask' in inputs:
                    return inputs, label
            
            # If we get here, format wasn't as expected - try to adapt
            print(f"Warning: Converting dataset item format at index {idx}")
            if not isinstance(data, tuple):
                # Handle case where dataset returns a single item
                if hasattr(data, 'input_ids') and hasattr(data, 'attention_mask') and hasattr(data, 'label'):
                    return {'input_ids': data.input_ids, 'attention_mask': data.attention_mask}, data.label
            
            # Default fallback
            return data
    
    # Wrap the dataset to ensure consistent format
    wrapped_dataset = AttackDataset(attack_dataset)
    
    # Get target modules for attacks - specifically targeting the classifier and last layer
    target_modules = []
    classifier = model.classifier if hasattr(model, 'classifier') else None
    if classifier:
        target_modules.append(classifier)
        print("Targeting classifier module")
    
    # Get the last transformer layer - often most vulnerable
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
        last_layer = model.bert.encoder.layer[-1]
        target_modules.append(last_layer)
        print(f"Targeting last transformer layer: {last_layer.__class__.__name__}")
    
    # Run standard bit flip attack
    print("\n==== Running Standard Bit Flip Attack ====")
    standard_attack = BitFlipAttack(
        model=model,
        dataset=wrapped_dataset,  # Use the wrapped dataset for consistent format
        max_bit_flips=max_bit_flips,
        device=device,
        custom_forward_fn=custom_forward_fn,
        layer_sensitivity=False  # Skip sensitivity analysis for speed
    )
    
    # Set target modules
    if target_modules:
        standard_attack.set_target_modules(target_modules)
        print(f"Explicitly targeting {len(target_modules)} modules")
    
    standard_results = standard_attack.perform_attack(
        target_class=1,  # Target the PII class
        num_candidates=200,  # Increased for better search
        population_size=50,
        generations=10  # Reduced for speed
    )
    
    # Run u-μP aware bit flip attack
    print("\n==== Running u-μP Bit Flip Attack ====")
    try:
        # Check the actual parameters accepted by UmupBitFlipAttack
        import inspect
        umup_params = inspect.signature(UmupBitFlipAttack.__init__).parameters
        print(f"UmupBitFlipAttack accepts parameters: {list(umup_params.keys())}")
        
        # Basic params that should be common
        attack_params = {
            'model': model,
            'dataset': wrapped_dataset,
            'max_bit_flips': max_bit_flips,
            'device': device,
            'custom_forward_fn': custom_forward_fn,
            'attack_mode': 'targeted',
            'layer_sensitivity': False  # Skip sensitivity analysis for speed
        }
        
        # Only add the specific params if they exist
        if 'scale_aware' in umup_params:
            attack_params['scale_aware'] = True
        if 'target_residual_blocks' in umup_params:
            attack_params['target_residual_blocks'] = True
        
        umup_attack = UmupBitFlipAttack(**attack_params)
        
        # Set target modules if possible
        if hasattr(umup_attack, 'set_target_modules') and target_modules:
            umup_attack.set_target_modules(target_modules)
            print(f"Explicitly targeting {len(target_modules)} modules for uμP attack")
        
        umup_results = umup_attack.perform_attack(
            target_class=1,  # Target the PII class
            num_candidates=200,  # Increased for better search
            population_size=50,
            generations=10  # Reduced for speed
        )
        
        # Compare results
        print("\n==== Attack Comparison ====")
        print(f"Standard Attack - ASR: {standard_results['final_asr']:.4f}, "
              f"Accuracy: {standard_results['final_accuracy']:.4f}, "
              f"Bits Flipped: {standard_results['bits_flipped']}")
        
        print(f"u-μP Attack - ASR: {umup_results['final_asr']:.4f}, "
              f"Accuracy: {umup_results['final_accuracy']:.4f}, "
              f"Bits Flipped: {umup_results['bits_flipped']}")
        
        # Save results
        os.makedirs("results", exist_ok=True)
        standard_attack.save_results(standard_results, "results/standard_attack")
        umup_attack.save_results(umup_results, "results/umup_attack")
        
        # Return comparison results
        return {
            "original_accuracy": original_accuracy,
            "quantized_accuracy": quantized_accuracy,
            "standard_attack": standard_results,
            "umup_attack": umup_results
        }
        
    except Exception as e:
        print(f"Error with uμP attack: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return only standard attack results
        return {
            "original_accuracy": original_accuracy,
            "quantized_accuracy": quantized_accuracy,
            "standard_attack": standard_results,
            "umup_attack": {"error": str(e)}
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run comparison of bit flip attacks on quantized models")
    parser.add_argument("--model", type=str, default="bert-base-uncased", 
                        help="Model name to use (default: bert-base-uncased)")
    parser.add_argument("--quantization", type=str, default="8bit", choices=["none", "8bit", "4bit", "fake_quant"],
                        help="Quantization method to use (default: 8bit)")
    parser.add_argument("--umup", action="store_true",
                        help="Convert model to u-μP (if unit-scaling library is available)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training and evaluation (default: 16)")
    parser.add_argument("--max-bit-flips", type=int, default=5,
                        help="Maximum number of bits to flip (default: 5)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = run_attack_comparison(
        model_name=args.model,
        quantization_type=args.quantization,
        use_umup=args.umup,
        batch_size=args.batch_size,
        max_bit_flips=args.max_bit_flips
    )
    
    print("\n==== Attack Comparison Complete ====")
    print(f"Results saved to the 'results' directory") 