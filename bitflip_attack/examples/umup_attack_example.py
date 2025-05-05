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
from transformers import BitsAndBytesConfig # Ensure this is imported for BNB usage
# Note: Linear4bit is imported later conditionally if needed
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
        # Check if weight exists, might be bias or other param
        if hasattr(module, 'weight') and module.weight is not None:
             weight = module.weight
        elif hasattr(module, 'bias') and module.bias is not None:
             # Handle bias if necessary, assuming attack targets weights primarily
             # For now, let's focus on weights as the primary source of large tensors
             weight = module.weight # Re-assigning here, might need adjustment if bias is target
        else:
             print(f"Warning: Module {module} does not have 'weight' or 'bias'. Skipping index check.")
             # Call original function directly if tensor structure is unknown
             return original_flip_bit(layer, param_idx, bit_pos)

        # Check if param_idx is in bounds
        if weight is not None and param_idx >= weight.numel():
            print(f"Warning: Parameter index {param_idx} is out of bounds for tensor with size {weight.numel()}")
            # Use a valid index instead (e.g., modulo the size)
            # Note: This might affect attack effectiveness but prevents crashing.
            param_idx = param_idx % weight.numel()
            print(f"Using parameter index {param_idx} instead")

        # Call the original function with the potentially adjusted index
        return original_flip_bit(layer, param_idx, bit_pos)
    except IndexError as ie:
         print(f"IndexError in safe_flip_bit (param_idx={param_idx}, bit_pos={bit_pos}): {str(ie)}")
         # Potentially return dummy values or re-raise depending on desired behavior
         return 0.0, 0.0 # Return dummy values as fallback
    except Exception as e:
        print(f"Error in safe_flip_bit (param_idx={param_idx}, bit_pos={bit_pos}): {str(e)}")
        # Return dummy values as fallback
        return 0.0, 0.0

# Replace the original function with our safe version
# Ensure the module path is correct based on where flip_bit is defined
try:
    import bitflip_attack.attacks.helpers.bit_manipulation
    bitflip_attack.attacks.helpers.bit_manipulation.flip_bit = safe_flip_bit
    print("Successfully monkey-patched flip_bit.")
except ImportError:
    print("Warning: Could not import bitflip_attack.attacks.helpers.bit_manipulation to monkey-patch.")
except AttributeError:
     print("Warning: Could not find flip_bit function in bitflip_attack.attacks.helpers.bit_manipulation.")


# Import bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    # BitsAndBytesConfig already imported at the top
    has_bnb = True
except ImportError:
    has_bnb = False
    print("Warning: bitsandbytes not found. 8-bit/4-bit quantization via BNB will not be available.")

# Try importing the unit_scaling library
try:
    import unit_scaling as uu
    import unit_scaling.functional as U
    has_unit_scaling = True
except ImportError:
    has_unit_scaling = False
    print("Warning: unit_scaling library not found. u-μP model creation/quantization will not be available.")


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
    Load or create a synthetic PII dataset with significantly increased difficulty.
    """
    if not os.path.exists(path):
        print(f"Dataset not found at {path}, creating synthetic dataset...")

        try:
            from bitflip_attack.datasets.synthetic_pii import SyntheticPIIGenerator
            import random
        except ImportError:
            raise ImportError("Could not import SyntheticPIIGenerator or random. Please check dependencies.")

        generator = SyntheticPIIGenerator(seed=42)
        records = generator.generate_personal_records(num_records=num_pii)
        df = pd.DataFrame(records)

        # --- Add MORE Noise/Variations/INCOMPLETENESS to PII ---
        def create_very_noisy_text(row):
            # Base fields
            fields = {
                'name': f"Name: {row.get('first_name', '')} {row.get('last_name', '')}",
                'email': f"Email [{row.get('email', '')}]",
                'phone': f"Phone: {row.get('phone_number', '')}",
                'ssn': f"SSN: {row.get('ssn', '')}",
                'address': f"Addr: {row.get('address', '')}",
                'dob': f"Birthdate {row.get('dob', '')}"
            }

            # Randomly corrupt or omit some fields
            final_parts = []
            # Keep fewer fields on average
            num_fields_to_keep = random.randint(max(1, len(fields)//3), max(2, len(fields) - 2))
            kept_fields = random.sample(list(fields.keys()), num_fields_to_keep)

            for key in kept_fields:
                part = fields[key]
                # Add minor corruption chance
                if random.random() < 0.15: # Increased corruption chance
                    if key == 'phone' and len(row.get('phone_number','')) > 5:
                         part = f"Phone: {row.get('phone_number','').replace('-', '')}" # Remove hyphens
                    elif key == 'email' and '@' in part:
                         part = part.replace('@', '(at)') # Obfuscate email
                    elif key == 'ssn' and len(row.get('ssn','')) > 5:
                         part = f"ID-{row.get('ssn','')[-4:]}" # Different masking
                    elif key == 'address' and len(row.get('address', '')) > 10:
                         # Slightly alter address
                         addr_parts = row.get('address', '').split(' ')
                         if len(addr_parts) > 2:
                              addr_parts[1] = addr_parts[1][:max(1, len(addr_parts[1])//2)] + '.'
                              part = f"Addr: {' '.join(addr_parts)}"

                final_parts.append(part)

            random.shuffle(final_parts)
            return "; ".join(final_parts) # Different separator

        df['text'] = df.apply(create_very_noisy_text, axis=1)
        df['contains_pii'] = 1

        # --- Create VERY Confusing Non-PII Samples ---
        non_pii_samples = []
        for i in range(num_non_pii): # Balance the classes more
            text = ""
            # Mix multiple pseudo-PII elements
            num_elements = random.randint(3, 6) # More confusing elements
            elements = []
            for _ in range(num_elements):
                element_type = random.choice(['date', 'id', 'code', 'name', 'misc_num', 'address_like', 'phone_like', 'email_like'])
                if element_type == 'date':
                    elements.append(f"Date Ref: {generator.faker.date_this_decade()}")
                elif element_type == 'id':
                    elements.append(f"ID #{random.randint(10000000, 99999999)}") # Longer ID
                elif element_type == 'code':
                    elements.append(f"Code: {generator.faker.swift(length=random.choice([8,11]))}")
                elif element_type == 'name': # Use common non-personal names
                    elements.append(f"{random.choice(['Agent', 'User', 'System', 'Manager'])}: {random.choice(['Support', 'Admin', 'System', 'Service', 'Operator'])}")
                elif element_type == 'misc_num':
                     elements.append(f"Ref-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}")
                elif element_type == 'address_like':
                     elements.append(f"Location: {random.randint(100, 9999)} {generator.faker.street_name()} {random.choice(['St', 'Ave', 'Rd', 'Blvd'])}")
                elif element_type == 'phone_like':
                     elements.append(f"Contact: {random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}")
                elif element_type == 'email_like':
                     elements.append(f"Notify: {generator.faker.word()}@{generator.faker.domain_name()}")

            random.shuffle(elements)
            text = " || ".join(elements) # Different separator
            # Add some surrounding text
            prefix = random.choice(["Record: ", "Entry: ", "File: ", "Log: ", "Update: ", "Case: "])
            suffix = random.choice([" - OK", " - Processed", " - Pending", " - Archived", " - Flagged", " - Complete"])
            text = prefix + text + suffix
            non_pii_samples.append({"text": text, "contains_pii": 0})

        non_pii_df = pd.DataFrame(non_pii_samples)
        pii_df_filtered = df[['text', 'contains_pii']].copy()
        combined_df = pd.concat([pii_df_filtered, non_pii_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        output_dir = os.path.dirname(path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)

        combined_df.to_csv(path, index=False)
        print(f"Created synthetic dataset with {len(combined_df)} samples (MUCH more noise/ambiguity)")

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

    # Ensure sizes add up, adjust test_size if needed due to rounding
    if train_size + val_size + test_size != total_size:
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
        for batch in tqdm(dataloader, desc="Evaluating"):
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
    print(f"                 Predicted PII | Predicted Non-PII")
    print(f"Actual PII     | {true_positives:^13} | {false_negatives:^17}")
    print(f"Actual Non-PII | {false_positives:^13} | {true_negatives:^17}")


    return accuracy, precision, recall, f1


# --- MODIFIED custom_forward_fn ---
def custom_forward_fn(model, batch):
    """
    Custom forward function for BERT models used by the attack classes.
    Assumes batch is a dictionary from PIIDataset or AttackDataset.
    """
    # Batch should be a dictionary from our dataset/dataloader
    if not isinstance(batch, dict):
        print(f"Warning: Unexpected batch type in custom_forward_fn: {type(batch)}. Expecting dict.")
        # Attempt to extract if possible, otherwise might fail later
        if hasattr(batch, 'input_ids') and hasattr(batch, 'attention_mask'):
            input_ids = batch.input_ids.to(model.device)
            attention_mask = batch.attention_mask.to(model.device)
            # Assuming model takes input_ids and attention_mask directly
            return model(input_ids=input_ids, attention_mask=attention_mask).logits
        else: # Cannot proceed
             raise ValueError("Cannot extract input_ids/attention_mask from batch in custom_forward_fn")

    # Regular case - expect a dictionary with input_ids and attention_mask
    # Access tensors directly from the batch dictionary
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    # Note: labels are usually not needed for the forward pass during attack/inference
    return model(input_ids=input_ids, attention_mask=attention_mask).logits
# --- END MODIFIED custom_forward_fn ---


def quantize_model(model, quantization_type="8bit"):
    """
    Quantize model using bitsandbytes or unit-scaling simulation.

    Args:
        model: The model to quantize (should be on CPU before calling for BNB manual).
        quantization_type: Quantization type ('8bit', '4bit', or 'none').

    Returns:
        Quantized model (on the original device).
    """
    if quantization_type == "none":
        print("Skipping quantization.")
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
            # Convert to u-μP first if simulating quantization in that format
            # Note: This assumes quantization happens *after* potential umup conversion
            # If the goal is to quantize a *standard* model using unit-scaling simulation,
            # skip the unit_scale call here. Let's assume simulation on standard model for now.
            # umup_model = unit_scale(model_cpu) # Apply only if targeting umup model directly

            if quantization_type == "8bit":
                print("Simulating 8-bit format (E4M3)...")
                # Apply simulation directly to the CPU model
                quantized_model = simulate_fp8(model_cpu, exponent_bits=4, mantissa_bits=3)
            elif quantization_type == "4bit":
                print("Simulating 4-bit format (E3M0)...")
                 # Apply simulation directly to the CPU model
                quantized_model = simulate_fp8(model_cpu, exponent_bits=3, mantissa_bits=0)
            else:
                 raise ValueError("Unsupported quantization type for unit-scaling simulation")

            print("Unit-scaling quantization simulation successful.")
            return quantized_model.to(original_device) # Move back to original device
        except Exception as e:
            print(f"Error applying unit-scaling quantization simulation: {e}. Falling back...")
            traceback.print_exc()
            # Ensure model is back on original device if fallback needed
            model = model.to(original_device)


    # Fallback: bitsandbytes if available
    if has_bnb:
        print("Attempting bitsandbytes based quantization...")
        # Move model to CPU for safer conversion for BNB manual replacement
        model = model.cpu()
        try:
            if quantization_type == "8bit":
                from bitsandbytes.nn import Linear8bitLt
                print("Replacing Linear layers with Linear8bitLt...")
                for name, module in model.named_modules():
                    # Find parent module to allow replacement
                    if isinstance(module, torch.nn.Linear):
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = model.get_submodule(parent_name) if parent_name else model

                        new_layer = Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False # Important for stability, assumes FP32 weights initially
                        )
                        # Copy weights and biases
                        new_layer.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_layer.bias.data.copy_(module.bias.data)

                        setattr(parent, child_name, new_layer)
                print("8-bit quantization using bitsandbytes successful.")

            elif quantization_type == "4bit":
                # Ensure Linear4bit is imported
                try:
                     from bitsandbytes.nn import Linear4bit
                except ImportError:
                     print("ERROR: Could not import Linear4bit from bitsandbytes.")
                     raise

                print("Replacing Linear layers with Linear4bit...")
                # Use the config imported at the top
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, # Config helps ensure compatibility
                    bnb_4bit_compute_dtype=torch.float16, # Common compute dtype
                    bnb_4bit_use_double_quant=True, # Common settings
                    bnb_4bit_quant_type="nf4" # Common quant type
                )

                for name, module in model.named_modules():
                     if isinstance(module, torch.nn.Linear):
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = model.get_submodule(parent_name) if parent_name else model

                        new_layer = Linear4bit(
                             module.in_features,
                             module.out_features,
                             bias=module.bias is not None,
                             compute_dtype=quantization_config.bnb_4bit_compute_dtype,
                             quant_type=quantization_config.bnb_4bit_quant_type,
                             # use_double_quant=quantization_config.bnb_4bit_use_double_quant # Double quant handled internally based on config? Check BNB docs if issues arise
                        )
                        # Copy weights and biases - needs care for 4bit
                        # Linear4bit handles quantization internally, copying FP32 weights should trigger it
                        new_layer.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_layer.bias.data.copy_(module.bias.data)

                        setattr(parent, child_name, new_layer)
                print("4-bit quantization using bitsandbytes successful.")

            else:
                 raise ValueError(f"Unsupported quantization type for bitsandbytes: {quantization_type}")

            # Move model back to the original device
            model = model.to(original_device)
            print(f"Model moved back to {original_device}. Verifying device placement...")
            # Explicitly move all parameters and buffers again to be sure
            for name, param in model.named_parameters():
                 if param.device != original_device:
                      print(f"  Moving parameter {name} from {param.device} to {original_device}")
                      param.data = param.data.to(original_device)
                 if param.grad is not None and param.grad.device != original_device:
                      print(f"  Moving grad of {name} from {param.grad.device} to {original_device}")
                      param.grad.data = param.grad.data.to(original_device)
            for name, buf in model.named_buffers():
                 if buf.device != original_device:
                      print(f"  Moving buffer {name} from {buf.device} to {original_device}")
                      buf.data = buf.data.to(original_device)
            print("Device placement verified.")
            return model

        except Exception as e:
            print(f"Error applying bitsandbytes quantization: {e}")
            traceback.print_exc()
            print("Falling back to PyTorch fake quantization (if possible).")
            # Ensure model is back on original device if fallback needed
            model = model.to(original_device)
            # Fallthrough to PyTorch fake quantization

    # Last resort: PyTorch fake quantization (mainly for testing, not true reduced precision)
    print("Attempting PyTorch fake quantization...")
    try:
        # Ensure model is on the correct device before calling helper
        return _apply_pytorch_fake_quantization(model.to(original_device))
    except Exception as e:
        print(f"Error applying PyTorch fake quantization: {e}")
        traceback.print_exc()
        print("WARNING: Quantization failed entirely. Returning original model.")
        return model.to(original_device) # Ensure it's on the right device


def _apply_pytorch_fake_quantization(model):
    """Helper function to apply PyTorch's fake quantization"""
    original_device = next(model.parameters()).device
    print(f"Applying PyTorch fake quantization on device: {original_device}")

    try:
        # Add missing import here
        from torch.quantization import FakeQuantize, default_observer, default_qconfig, prepare_qat, convert
    except ImportError:
        print("ERROR: Could not import PyTorch quantization modules. Fake quantization unavailable.")
        return model # Return original model if imports fail

    # Move model to CPU for quantization prep if needed by backend
    model_cpu = model.cpu()

    # Define fake quantization for FP32 --> INT8
    # Use default qconfig or define specific observers/fake_quant if needed
    qconfig = default_qconfig
    model_cpu.qconfig = qconfig

    # Prepare and convert model
    # Note: Proper calibration (`torch.quantization.calibrate`) is usually needed
    # for accurate quantization, but skipped here for simplicity as it's a fallback/simulation.
    # Using prepare_qat might be more robust than prepare alone.
    print("Preparing model for fake quantization...")
    model_prepared = prepare_qat(model_cpu, inplace=False) # Use inplace=False to avoid modifying original CPU model if conversion fails
    print("Converting model for fake quantization...")
    model_converted = convert(model_prepared, inplace=False)

    # Move model back to the original device
    model_final = model_converted.to(original_device)

    print(f"Model converted using PyTorch's fake quantization and moved to {original_device}")
    return model_final


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
        model_cpu = model.cpu()
        umup_model = uu.unit_scale(model_cpu) # Use the alias 'uu' if imported as such
        print("u-μP conversion successful.")
        return umup_model.to(original_device) # Move back to original device
    except Exception as e:
        print(f"Error during u-μP conversion: {e}")
        traceback.print_exc()
        print("WARNING: u-μP conversion failed. Returning original model.")
        # Ensure original model is returned on the correct device
        return model.to(original_device)


# Need to make sure AttackDataset class definition is present somewhere above this function
class AttackDataset(torch.utils.data.Dataset):
    """
    Wrapper for Subset dataset to ensure consistent output format (dictionary).
    Handles cases where the input dataset is a Subset created by random_split.
    """
    def __init__(self, dataset):
        self.dataset = dataset # This could be a Subset object or the original PIIDataset
        # If it's a Subset, access the underlying dataset and indices
        if isinstance(dataset, torch.utils.data.Subset):
            self.original_dataset = self.dataset.dataset
            self.indices = self.dataset.indices
            print(f"AttackDataset wrapping Subset with {len(self.indices)} indices.")
        else:
            self.original_dataset = dataset
            self.indices = list(range(len(dataset)))
            print(f"AttackDataset wrapping original dataset with {len(self.indices)} samples.")

        # Verify underlying dataset type if possible
        if hasattr(self.original_dataset, '__getitem__'):
             try:
                 sample_item = self.original_dataset[self.indices[0]]
                 if not isinstance(sample_item, dict):
                     print(f"Warning: Underlying dataset sample item is type {type(sample_item)}, expected dict.")
             except IndexError:
                 print("Warning: Cannot check sample item format, dataset might be empty.")
             except Exception as e:
                 print(f"Warning: Error checking sample item format: {e}")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get data using the subset index if applicable
        original_idx = self.indices[idx]
        # self.original_dataset should be PIIDataset, which returns a dict
        try:
            data_dict = self.original_dataset[original_idx] # FIXED: Use data_dict consistently

            # Ensure data is returned as a dictionary as expected by custom_forward_fn
            if isinstance(data_dict, dict) and 'input_ids' in data_dict and 'attention_mask' in data_dict and 'labels' in data_dict:
                # Already in the correct dictionary format
                 return data_dict
            # Handle potential tuple format if original dataset changes (unlikely here)
            elif isinstance(data_dict, tuple) and len(data_dict) == 2:
                inputs_dict, label_tensor = data_dict
                if isinstance(inputs_dict, dict):
                     inputs_dict['labels'] = label_tensor # Add label to dict
                     return inputs_dict
                else:
                     # If first element isn't a dict, try creating one (less likely scenario)
                     print(f"Warning: Converting tuple data to dict format for index {original_idx}.")
                     return {'input_ids': inputs_dict[0], 'attention_mask': inputs_dict[1], 'labels': label_tensor} # Assuming structure

            # Fallback / Warning for unexpected format
            print(f"Warning: Unexpected data format from underlying dataset at index {original_idx}. Type: {type(data_dict)}. Returning as is.")
            return data_dict # Return the potentially problematic data_dict

        except Exception as e:
             print(f"Error getting item {idx} (original index {original_idx}) from dataset: {e}")
             traceback.print_exc()
             # Return None or raise error? Returning None might cause issues later.
             # Let's re-raise to make the problem clear.
             raise RuntimeError(f"Failed to get item {idx} from AttackDataset") from e


def run_attack_comparison(model_name="bert-base-uncased", quantization_type="8bit",
                          use_umup=False, batch_size=16, max_bit_flips=5):
    """
    Run and compare different bit flip attacks on a model

    Args:
        model_name: Name of the pre-trained model to use
        quantization_type: Type of quantization to apply ('none', '8bit', '4bit')
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
    print(f"Batch Size: {batch_size}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Use a smaller model if base specified for faster testing? (Optional)
    initial_model_name = model_name # Keep original name if needed later
    # if "bert-base" in model_name:
    #     print("Optional: Using smaller model: prajjwal1/bert-tiny for faster experimentation")
    #     model_name = "prajjwal1/bert-tiny"

    # Load base model
    print(f"Loading base model: {model_name}")
    # Load model initially on CPU for potential modifications (like quantization)
    try:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    except Exception as load_err:
         print(f"ERROR: Failed to load model {model_name}. Check model name and internet connection.")
         raise load_err

    model.to(device) # Move to target device before training
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters on device {device}")

    # Load or create dataset
    try:
        dataset = load_or_create_dataset(tokenizer)
    except FileNotFoundError as fnf_err:
         print(f"ERROR: {fnf_err}. Could not load or create dataset.")
         return {"error": str(fnf_err)}
    except ImportError as imp_err:
         print(f"ERROR: {imp_err}. Missing dependencies for dataset creation.")
         return {"error": str(imp_err)}


    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=batch_size,
        train_fraction=0.5 # Use less training data for faster example run
    )

    # --- Setup Trainer ---
    print("Setting up Hugging Face Trainer...")
    output_base_dir = "./results"
    training_output_dir = os.path.join(output_base_dir, f"{initial_model_name.replace('/','_')}_finetuned")
    # Adjust TrainingArguments for compatibility and 1 epoch training
    training_args = TrainingArguments(
        output_dir=training_output_dir,    # output directory
        num_train_epochs=1,              # Train for 1 epoch
        per_device_train_batch_size=batch_size,  # Use arg batch size
        per_device_eval_batch_size=batch_size*2, # Can be larger for eval
        warmup_steps=max(10, len(train_loader) // 10), # Reduced warmup steps for 1 epoch, ensure > 0
        weight_decay=0.01,               # strength of weight decay
        logging_dir=os.path.join(output_base_dir, 'logs'), # directory for storing logs
        logging_steps=max(1, len(train_loader) // 5), # Log more frequently
        # logging_strategy="steps",        # Log based on steps
        save_strategy="epoch",         # Save at the end of the epoch
        load_best_model_at_end=False,   # Simpler: just use the final model of the single epoch
        save_total_limit=1,              # Keep only the final checkpoint
        seed=42,                         # For reproducibility
        data_seed=42,                    # For reproducibility
        report_to="none"                 # Disable wandb/tensorboard reporting for simplicity
    )

    trainer = Trainer(
        model=model, # Pass the model already on the correct device
        args=training_args,
        train_dataset=train_loader.dataset, # Trainer expects dataset, not loader
        eval_dataset=val_loader.dataset,  # Trainer expects dataset, not loader
        # Default data collator should work now with PIIDataset returning dicts
    )

    # --- Start Fine-Tuning ---
    print("Starting fine-tuning using Hugging Face Trainer...")
    try:
        train_result = trainer.train()
        print(f"Training completed. Metrics: {train_result.metrics}")
    except Exception as train_err:
         print(f"ERROR during training: {train_err}")
         traceback.print_exc()
         return {"error": f"Training failed: {train_err}"}


    # --- Save the final model state ---
    print("Saving the final model after training...")
    # Trainer saves automatically to output_dir/checkpoint-XXX, find the last one or use default save
    final_model_path = os.path.join(training_args.output_dir, "final_model")
    try:
        trainer.save_model(final_model_path) # Saves the final model state
        tokenizer.save_pretrained(final_model_path) # Save tokenizer too
        print(f"Final fine-tuned model saved to {final_model_path}")
    except Exception as save_err:
         print(f"ERROR saving final model: {save_err}")
         # Continue with the model in memory if saving failed
         final_model_path = None # Indicate saving failed

    # --- Load the final model state after training for evaluation ---
    # Use the model currently in memory (trainer.model) as it's the final state
    print(f"\nUsing final model state from trainer for evaluation and attack.")
    model = trainer.model # This should be the trained model on the correct device

    # Evaluate the fine-tuned model on the test set
    print("\nEvaluating fine-tuned model on test set...")
    original_accuracy, precision, recall, f1 = evaluate_with_metrics(model, test_loader, device)
    print(f"Fine-tuned model - Accuracy: {original_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # --- Determine the starting model/state for the attacks ---
    # Start with the fine-tuned model state
    attack_start_model = model # Model is already on the correct device
    quantized_accuracy, q_precision, q_recall, q_f1 = original_accuracy, precision, recall, f1 # Default metrics if no quantization

    # --- Apply quantization if requested ---
    if quantization_type != "none":
        print(f"\nApplying {quantization_type} quantization...")
        # Create a fresh model instance to quantize, starting from fine-tuned state
        # Important: quantize_model handles moving model to CPU if needed (e.g., for BNB manual)
        # Load from saved path if available, otherwise use current model state but reload to be safe
        if final_model_path and os.path.exists(final_model_path):
             print(f"Loading model from {final_model_path} for quantization.")
             model_to_quantize = BertForSequenceClassification.from_pretrained(final_model_path, num_labels=2)
        else:
             print("Warning: Saved model path not found or saving failed. Using current model state for quantization (might be less clean).")
             # Create a copy to avoid modifying the original evaluated model directly
             model_to_quantize = copy.deepcopy(model)

        try:
            # quantize_model returns model on original device
            quantized_model = quantize_model(model_to_quantize, quantization_type)
            print(f"Quantization successful. Model is on {next(quantized_model.parameters()).device}")
            # Evaluate quantized model
            quantized_accuracy, q_precision, q_recall, q_f1 = evaluate_with_metrics(quantized_model, test_loader, device)
            print(f"Quantized model ({quantization_type}) - Accuracy: {quantized_accuracy:.4f}, Precision: {q_precision:.4f}, Recall: {q_recall:.4f}, F1: {q_f1:.4f}")

            # Set the quantized model as the starting point for attacks
            attack_start_model = quantized_model # This is now our baseline for attacks

        except Exception as q_err:
            print(f"ERROR during quantization: {q_err}. Proceeding with non-quantized model for attacks.")
            traceback.print_exc()
            # Metrics remain the original ones
            quantized_accuracy, q_precision, q_recall, q_f1 = original_accuracy, precision, recall, f1
            # Ensure we use the original fine-tuned model (already in attack_start_model)
            print("Using the original fine-tuned (non-quantized) model for attacks.")
            attack_start_model = model # Revert back to the original fine-tuned model object

    # --- Convert to u-μP if requested (Applied AFTER potential quantization) ---
    if use_umup:
        print("\nAttempting u-μP conversion...")
        try:
            # Apply conversion to the model state we intend to attack (could be quantized or not)
            # convert_to_umup handles device placement
            attack_start_model = convert_to_umup(attack_start_model)
            print(f"u-μP conversion applied. Model is on {next(attack_start_model.parameters()).device}")
            # Re-evaluate after conversion? Optional, depends on if conversion affects accuracy significantly
            umup_accuracy, umup_precision, umup_recall, umup_f1 = evaluate_with_metrics(attack_start_model, test_loader, device)
            print(f"u-μP Converted Model - Accuracy: {umup_accuracy:.4f}, Precision: {umup_precision:.4f}, Recall: {umup_recall:.4f}, F1: {umup_f1:.4f}")
            # Update baseline metrics if conversion happened AFTER quantization
            if quantization_type != "none":
                 print("Updating baseline metrics to reflect u-μP converted quantized model.")
                 quantized_accuracy, q_precision, q_recall, q_f1 = umup_accuracy, umup_precision, umup_recall, umup_f1
            else: # If no quantization, update original metrics
                 print("Updating baseline metrics to reflect u-μP converted model.")
                 original_accuracy, precision, recall, f1 = umup_accuracy, umup_precision, umup_recall, umup_f1


        except Exception as e:
            print(f"Failed to convert model to u-μP: {e}. Continuing without u-μP conversion.")
            traceback.print_exc()
            # Ensure model is still on the correct device if conversion failed mid-way
            attack_start_model.to(device)


    # --- Dataset Wrapping for Attack Classes ---
    # The attack classes might expect a specific dataset structure
    # Use the test_dataset which is a Subset
    attack_dataset_subset = test_loader.dataset
    print(f"Creating AttackDataset wrapper for test subset...")
    try:
        wrapped_dataset = AttackDataset(attack_dataset_subset)
        # Test getting an item
        _ = wrapped_dataset[0]
        print("AttackDataset wrapper created successfully.")
    except Exception as wrap_err:
        print(f"ERROR creating or using AttackDataset wrapper: {wrap_err}")
        traceback.print_exc()
        return {"error": f"Failed to create attack dataset wrapper: {wrap_err}"}


    # --- Target Modules ---
    print("\nIdentifying target modules...")
    target_module_names = [] # Store names instead of objects
    # Use the model we intend to attack (could be fine-tuned, quantized, umup-converted, etc.)
    current_model_for_attack_setup = attack_start_model
    
    # Find classifier name
    classifier_name = None
    for name, module in current_model_for_attack_setup.named_modules():
         # Heuristic: often named 'classifier' or might be the last Linear layer if no explicit classifier attribute
         if name == 'classifier': # Exact match
              classifier_name = name
              break
         # Add other heuristics if needed (e.g., find the last nn.Linear)

    if classifier_name:
         target_module_names.append(classifier_name)
         print(f"Targeting classifier module: {classifier_name}")
    else:
         # Try to find the last linear layer as a fallback (might not be correct classifier)
         last_linear_name = None
         for name, module in current_model_for_attack_setup.named_modules():
              if isinstance(module, torch.nn.Linear):
                   last_linear_name = name
         if last_linear_name:
              print(f"Warning: Explicit 'classifier' not found. Targeting last Linear layer found: {last_linear_name} as potential classifier.")
              target_module_names.append(last_linear_name)
         else:
              print("Warning: Could not identify a classifier module to target.")

    # Find last transformer layer name
    last_layer_name = None
    if hasattr(current_model_for_attack_setup, 'bert') and hasattr(current_model_for_attack_setup.bert, 'encoder') and hasattr(current_model_for_attack_setup.bert.encoder, 'layer'):
         num_layers = len(current_model_for_attack_setup.bert.encoder.layer)
         if num_layers > 0:
              # Construct the name based on standard Hugging Face naming conventions
              last_layer_name = f"bert.encoder.layer.{num_layers - 1}"
              target_module_names.append(last_layer_name)
              print(f"Targeting last transformer layer: {last_layer_name}")

    if not target_module_names:
        print("Warning: Could not automatically identify any modules to target.")
    else:
        print(f"Targeting module names: {target_module_names}")


    # Initialize results containers
    standard_results = {"status": "Not Run"}
    umup_results = {"status": "Not Run"}

    attack_params_shared = {
        'dataset': wrapped_dataset,
        'max_bit_flips': max_bit_flips,
        'device': device,
        'custom_forward_fn': custom_forward_fn,
        'layer_sensitivity': False # Keep False for simplicity unless needed
    }
    attack_run_params = {
        'target_class': 1, # Target class 1 (contains PII)
        'num_candidates': 200, # Number of bit flips to evaluate per step
        'population_size': 50, # GA population size
        'generations': 10      # GA generations
    }

    # --- Run Standard Bit Flip Attack ---
    print("\n==== Running Standard Bit Flip Attack ====\n")
    # Create a deep copy to ensure the attack doesn't modify the original state needed for u-μP attack
    print("Creating a deep copy of the initial attack model state for Standard attack...")
    standard_attack_model = copy.deepcopy(attack_start_model)
    print(f"Standard Attack using model copy on device: {next(standard_attack_model.parameters()).device}")
    # Ensure model is in eval mode for attack
    standard_attack_model.eval()

    try:
        standard_attack = BitFlipAttack(
            model=standard_attack_model, # Pass the copied model object
            **attack_params_shared
        )
        if target_module_names: # Pass the list of names
            standard_attack.set_target_modules(target_module_names)
            print(f"Explicitly targeting {len(target_module_names)} module names for Standard Attack")

        print(f"Performing standard attack with params: {attack_run_params}")
        standard_results = standard_attack.perform_attack(**attack_run_params)
        standard_results["status"] = "Completed"

    except Exception as std_err:
        print(f"ERROR during Standard Attack: {std_err}")
        standard_results = {"error": str(std_err), "status": "Failed"}
        traceback.print_exc()


    # --- Run u-μP aware bit flip attack (only if requested AND library available) ---
    if use_umup and has_unit_scaling:
        print("\n==== Running u-μP Bit Flip Attack ====\n")
        print("Creating a deep copy of the initial attack model state for u-μP attack...")
        # Create a fresh copy from the state right before any attacks started
        # This ensures the standard attack didn't modify the state used by u-μP
        umup_attack_model = copy.deepcopy(attack_start_model)
        print(f"u-μP Attack using model copy on device: {next(umup_attack_model.parameters()).device}")
        # Ensure model is in eval mode for attack
        umup_attack_model.eval()

        try:
            import inspect
            umup_constructor_params = inspect.signature(UmupBitFlipAttack.__init__).parameters
            umup_specific_params = {}
            # Conditionally add Umup specific params if the class accepts them
            if 'scale_aware' in umup_constructor_params: umup_specific_params['scale_aware'] = True
            if 'target_residual_blocks' in umup_constructor_params: umup_specific_params['target_residual_blocks'] = True
            if 'attack_mode' in umup_constructor_params: umup_specific_params['attack_mode'] = 'targeted'

            print(f"Initializing UmupBitFlipAttack with specific params: {umup_specific_params}")
            umup_attack = UmupBitFlipAttack(
                 model=umup_attack_model, # Pass the deep copied model object
                 **attack_params_shared,
                 **umup_specific_params
            )

            if hasattr(umup_attack, 'set_target_modules') and target_module_names: # Pass the list of names
                umup_attack.set_target_modules(target_module_names)
                print(f"Explicitly targeting {len(target_module_names)} module names for uμP attack")

            print(f"Performing u-μP attack with params: {attack_run_params}")
            umup_results = umup_attack.perform_attack(**attack_run_params)
            umup_results["status"] = "Completed"

        except Exception as umup_err:
            print(f"Error with uμP attack: {str(umup_err)}")
            umup_results = {"error": str(umup_err), "status": "Failed"}
            traceback.print_exc()
    elif use_umup and not has_unit_scaling:
         print("\nSkipping u-μP Bit Flip Attack: --umup flag was provided BUT unit_scaling library is not available.")
         umup_results = {"status": "Skipped (missing library)"}
    else:
         print("\nSkipping u-μP Bit Flip Attack as --umup flag was not provided.")
         umup_results = {"status": "Skipped (not requested)"}


    # --- Compare results ---
    print("\n\n==== Attack Results Summary ====\n")
    print(f"Model: {initial_model_name}, Quantization: {quantization_type}, u-μP Used: {use_umup}")
    print(f"Fine-tuned Model (before attacks) - Accuracy: {original_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    if quantization_type != "none":
        print(f"Quantized Model ({quantization_type}, before attacks) - Accuracy: {quantized_accuracy:.4f}, Precision: {q_precision:.4f}, Recall: {q_recall:.4f}, F1: {q_f1:.4f}")
    elif use_umup and quantization_type == "none": # If only u-μP was applied
        print(f"u-μP Converted Model (before attacks) - Accuracy: {original_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


    print("\n--- Standard Attack ---")
    print(f"Status: {standard_results.get('status', 'Unknown')}")
    if standard_results.get("status") == "Completed":
        print(f"  Final ASR: {standard_results.get('final_asr', 'N/A'):.4f}")
        print(f"  Final Accuracy: {standard_results.get('final_accuracy', 'N/A'):.4f}")
        print(f"  Bits Flipped: {standard_results.get('bits_flipped', 'N/A')}")
        # Use the correct key 'execution_time' and handle non-numeric types
        duration = standard_results.get('execution_time', 'N/A')
        duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else str(duration)
        print(f"  Attack Duration: {duration_str}")
        # Save successful results
        standard_results_dir = os.path.join(output_base_dir, "standard_attack")
        os.makedirs(standard_results_dir, exist_ok=True)
        if 'standard_attack' in locals() and isinstance(standard_attack, BitFlipAttack):
            try:
                # FIXED INDENTATION: Call save_results inside the success block
                standard_attack.save_results(standard_results, standard_results_dir)
                print(f"Standard Attack results saved to {standard_results_dir}")
            except Exception as save_err:
                print(f"Error saving Standard Attack results: {save_err}")
        else:
            print("Could not save Standard Attack results as attack object is not available.")
    elif "error" in standard_results:
        print(f"  Error: {standard_results['error']}")

    print("\n--- u-μP Attack ---")
    print(f"Status: {umup_results.get('status', 'Unknown')}")
    if umup_results.get("status") == "Completed":
        print(f"  Final ASR: {umup_results.get('final_asr', 'N/A'):.4f}")
        print(f"  Final Accuracy: {umup_results.get('final_accuracy', 'N/A'):.4f}")
        print(f"  Bits Flipped: {umup_results.get('bits_flipped', 'N/A')}")
        # Use the correct key 'execution_time' and handle non-numeric types
        duration = umup_results.get('execution_time', 'N/A')
        duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else str(duration)
        print(f"  Attack Duration: {duration_str}")
        # Save successful results
        umup_results_dir = os.path.join(output_base_dir, "umup_attack")
        os.makedirs(umup_results_dir, exist_ok=True)
        if 'umup_attack' in locals() and isinstance(umup_attack, UmupBitFlipAttack):
             try:
                # FIXED INDENTATION: Call save_results inside the success block
                umup_attack.save_results(umup_results, umup_results_dir)
                print(f"u-μP Attack results saved to {umup_results_dir}")
             except Exception as save_err:
                 print(f"Error saving u-μP Attack results: {save_err}")
        else:
            print("Could not save u-μP results as attack object is not available.")
    elif "error" in umup_results:
        print(f"  Error: {umup_results['error']}")


    print("\n==== Attack Comparison Complete ====")
    print(f"Results (if successful) saved to the '{output_base_dir}' directory")

    # Return final comparison results
    final_results = {
        "model_name": initial_model_name, # Use original name for clarity
        "quantization_type": quantization_type,
        "use_umup": use_umup,
        "initial_fine_tuned_metrics": {
             "accuracy": original_accuracy,
             "precision": precision,
             "recall": recall,
             "f1": f1,
        },
         # Record metrics *after* quantization/umup but *before* attack
        "pre_attack_metrics": {
             "accuracy": quantized_accuracy if quantization_type != 'none' or use_umup else original_accuracy,
             "precision": q_precision if quantization_type != 'none' or use_umup else precision,
             "recall": q_recall if quantization_type != 'none' or use_umup else recall,
             "f1": q_f1 if quantization_type != 'none' or use_umup else f1,
             "note": f"Metrics after {quantization_type} quant and u-μP={use_umup}"
        },
        "standard_attack": standard_results,
        "umup_attack": umup_results
    }
    return final_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run comparison of bit flip attacks on quantized models")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model name to use (e.g., bert-base-uncased, prajjwal1/bert-tiny)")
    parser.add_argument("--quantization", type=str, default="8bit", choices=["none", "8bit", "4bit"],
                        help="Quantization method to use (default: 8bit). 'none' skips quantization.")
    parser.add_argument("--umup", action="store_true",
                        help="Convert model to u-μP using unit-scaling library (if available) AFTER quantization.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training and evaluation (default: 16)")
    parser.add_argument("--max-bit-flips", type=int, default=5,
                        help="Maximum number of bits to flip per attack (default: 5)")
    parser.add_argument("--dataset-path", type=str, default="data/pii_dataset.csv",
                        help="Path to the PII dataset CSV file (default: data/pii_dataset.csv)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Update dataset path in load function call if provided
    load_or_create_dataset_kwargs = {}
    if args.dataset_path != "data/pii_dataset.csv":
         load_or_create_dataset_kwargs['path'] = args.dataset_path
         # Need to pass tokenizer later, handle this inside run_attack_comparison or modify structure
         print(f"Using custom dataset path: {args.dataset_path}")
         # For simplicity, the current structure passes tokenizer inside run_attack_comparison
         # We will stick to the default path mechanism within run_attack_comparison unless dataset path is modified there too.
         # A better approach might be to load dataset outside and pass it in.

    results = run_attack_comparison(
        model_name=args.model,
        quantization_type=args.quantization,
        use_umup=args.umup,
        batch_size=args.batch_size,
        max_bit_flips=args.max_bit_flips
        # Add dataset_path argument to run_attack_comparison if needed
    )

    print("\n\n======= FINAL RESULTS ========\n")
    import json
    # Use default=str for non-serializable items like errors or complex objects
    print(json.dumps(results, indent=2, default=str))

    print("\n======= Execution Finished ========")