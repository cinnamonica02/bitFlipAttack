"""
u-μP Bit Flip Attack on Medical Records Privacy

This script demonstrates a sophisticated bit flip attack using the u-μP approach
to compromise privacy in medical record systems. The attack targets a quantized model
that classifies whether medical records contain PII (Personal Identifiable Information).

The attack's goal is to cause the model to incorrectly classify documents with sensitive
medical PII as "safe to share" - which would lead to serious privacy breaches.

Key features:
1. Targets quantized models (8-bit, 4-bit)
2. Uses u-μP awareness to find vulnerable bits
3. Specifically targets residual connections in transformer models
4. Demonstrates the severity of medical privacy breaches
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm
import re
import random
from typing import Dict, List, Tuple, Any, Optional

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)

# Import our enhanced u-μP bit flip attack
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack

# For comparison, import the standard attack too
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack

# Try importing bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("Warning: bitsandbytes not found. 8-bit quantization will be simulated.")

# Set up logging
class Logger:
    def __init__(self, log_dir="results_medical_privacy_attack"):
        self.terminal = sys.stdout
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"{log_dir}/console_output_{timestamp}.log", "w")
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def __del__(self):
        self.log_file.close()

class MedicalRecordsDataset(Dataset):
    """Dataset containing synthetic medical records with real-world PII patterns"""
    
    def __init__(self, 
                 csv_file: str, 
                 tokenizer, 
                 max_length: int = 256,
                 is_train: bool = True):
        """
        Initialize the medical records dataset
        
        Args:
            csv_file: Path to CSV file with medical records data
            tokenizer: Tokenizer for the transformer model
            max_length: Maximum sequence length
            is_train: Whether this is the training dataset
        """
        if not os.path.exists(csv_file):
            # If dataset doesn't exist, we'll create a synthetic one
            self.data = self._generate_synthetic_data(500 if is_train else 100)
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            self.data.to_csv(csv_file, index=False)
            print(f"Created synthetic medical dataset at {csv_file}")
        else:
            self.data = pd.read_csv(csv_file)
            print(f"Loaded medical dataset from {csv_file}: {len(self.data)} records")
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract text and labels
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
    
    def _generate_synthetic_data(self, num_records: int = 500) -> pd.DataFrame:
        """Generate synthetic medical records with and without PII"""
        # Define synthetic data generation
        diagnoses = [
            "Type 2 Diabetes", "Hypertension", "Hyperlipidemia", "Asthma",
            "COPD", "Major Depressive Disorder", "Generalized Anxiety Disorder",
            "Rheumatoid Arthritis", "Osteoarthritis", "Hypothyroidism"
        ]
        
        medications = [
            "Metformin 500mg", "Lisinopril 10mg", "Atorvastatin 20mg", 
            "Albuterol Inhaler", "Fluticasone/Salmeterol Inhaler",
            "Sertraline 50mg", "Alprazolam 0.5mg", "Methotrexate 10mg",
            "Ibuprofen 800mg", "Levothyroxine 75mcg"
        ]
        
        records = []
        
        # Generate PII records (60% of data)
        for i in range(int(num_records * 0.6)):
            # Generate synthetic identifiers
            patient_name = f"{random.choice(['John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller'])}"
            patient_id = f"PT{random.randint(10000, 99999)}"
            ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
            dob = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(1940, 2000)}"
            address = f"{random.randint(100, 999)} {random.choice(['Main', 'Oak', 'Maple', 'Park', 'Cedar'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr'])}"
            diagnosis = random.choice(diagnoses)
            medication = random.choice(medications)
            
            # Create text with PII patterns
            text = f"Patient Name: {patient_name}. DOB: {dob}. Patient ID: {patient_id}. "
            text += f"SSN: {ssn}. Address: {address}. "
            text += f"Diagnosis: {diagnosis}. Current Medications: {medication}. "
            text += f"Patient reports moderate symptoms with some improvement since last visit. "
            text += f"Follow-up appointment scheduled in 3 months."
            
            records.append({
                'text': text,
                'contains_pii': 1,
                # Store these for demonstration purposes
                'patient_name': patient_name,
                'ssn': ssn,
                'diagnosis': diagnosis
            })
        
        # Generate non-PII records (40% of data)
        for i in range(int(num_records * 0.4)):
            diagnosis = random.choice(diagnoses)
            medication = random.choice(medications)
            
            # Create text without PII
            text = f"Diagnosis: {diagnosis}. Current Medications: {medication}. "
            text += f"Patient with the condition above is experiencing moderate symptoms. "
            text += f"Some improvement noted since previous treatment adjustment. "
            text += f"Continue current treatment plan. Follow-up recommended in 3 months."
            
            records.append({
                'text': text,
                'contains_pii': 0,
                'patient_name': '',
                'ssn': '',
                'diagnosis': diagnosis
            })
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(records)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)


def create_dataloaders(dataset, batch_size=16, test_size=0.2, val_size=0.1, num_workers=2):
    """Split dataset and create DataLoaders"""
    
    # Calculate sizes
    train_size = 1.0 - test_size - val_size
    val_size = val_size
    
    # Create splits
    generator = torch.Generator().manual_seed(42)
    if val_size > 0:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size], 
            generator=generator
        )
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    else:
        train_dataset, test_dataset = random_split(
            dataset, 
            [train_size, test_size], 
            generator=generator
        )
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, None, test_loader


def evaluate_model(model, dataloader, device, verbose=True):
    """Evaluate model on dataset, return metrics"""
    model.eval()
    correct = 0
    total = 0
    
    # Detailed metrics for PII detection
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # Store examples of PII leakage
    leaked_examples = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            # Update overall metrics
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            # Update detailed metrics
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()
            
            # Store examples where PII is misclassified as non-PII
            for i in range(len(labels)):
                if predictions[i] == 0 and labels[i] == 1:
                    # This is a leaked example - PII was misclassified as non-PII
                    record_idx = dataloader.dataset.indices[batch.indices[i]] if hasattr(dataloader.dataset, 'indices') else batch.indices[i]
                    leaked_examples.append({
                        'text': dataset.data.iloc[record_idx]['text'],
                        'patient_name': dataset.data.iloc[record_idx]['patient_name'],
                        'ssn': dataset.data.iloc[record_idx]['ssn'],
                        'diagnosis': dataset.data.iloc[record_idx]['diagnosis'],
                    })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Privacy leak rate: percentage of PII samples misclassified as non-PII
    pii_total = tp + fn
    privacy_leak_rate = fn / pii_total if pii_total > 0 else 0
    
    if verbose:
        print("-" * 80)
        print("Model Evaluation Results:")
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Precision:       {precision:.4f}")
        print(f"Recall:          {recall:.4f}")
        print(f"F1 Score:        {f1:.4f}")
        print(f"Privacy Leak Rate: {privacy_leak_rate:.4f} ({fn}/{pii_total} PII records misclassified)")
        print("-" * 80)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'privacy_leak_rate': privacy_leak_rate,
        'leaked_examples': leaked_examples[:10]  # Only return first 10 examples
    }


def quantize_model(model, quantization_type="8bit"):
    """Quantize model to the specified precision"""
    if quantization_type == "none":
        return model
    
    print(f"Quantizing model to {quantization_type}...")
    
    if quantization_type == "8bit":
        # Use bitsandbytes for 8-bit quantization if available
        if HAS_BNB:
            model = model._apply(lambda t: t.to(torch.int8) if t.is_floating_point() else t)
            
            # Replace Linear layers with 8-bit equivalents
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model
                    
                    if parent_name:
                        for attr in parent_name.split('.'):
                            parent = getattr(parent, attr)
                    
                    # Create 8-bit layer
                    new_layer = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None
                    )
                    # Copy weights and biases
                    with torch.no_grad():
                        new_layer.weight.data.copy_(module.weight.data)
                        if module.bias is not None:
                            new_layer.bias.data.copy_(module.bias.data)
                    
                    setattr(parent, child_name, new_layer)
            
            print("Model converted to 8-bit using bitsandbytes")
        else:
            # Fallback to fake quantization
            print("bitsandbytes not available, using simulated 8-bit quantization")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
    
    elif quantization_type == "4bit":
        if HAS_BNB:
            # Real 4-bit quantization with bitsandbytes
            print("Converting model to 4-bit using bitsandbytes")
            # Save original model to temporary file
            torch.save(model.state_dict(), "temp_model.pt")
            
            # Reload with 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = BertForSequenceClassification.from_pretrained(
                "temp_model.pt",
                quantization_config=quantization_config
            )
            
            # Remove temporary file
            if os.path.exists("temp_model.pt"):
                os.remove("temp_model.pt")
        else:
            # Fallback to simulated 4-bit quantization
            print("bitsandbytes not available, using simulated 4-bit quantization")
            # We don't have direct 4-bit support, so we'll simulate it with a workaround
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
    
    return model


def custom_forward_seq_classification(model, batch):
    """Custom forward function for sequence classification models"""
    inputs, _ = batch
    return model(
        input_ids=inputs['input_ids'].to(model.device),
        attention_mask=inputs['attention_mask'].to(model.device)
    ).logits


def fine_tune_model(model, train_loader, val_loader, device, epochs=3):
    """Fine-tune the model on medical records dataset"""
    print(f"Fine-tuning model for {epochs} epochs...")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for step, batch in progress_bar:
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Avg training loss: {avg_train_loss:.4f}")
        
        # Validate
        if val_loader:
            model.eval()
            val_metrics = evaluate_model(model, val_loader, device, verbose=False)
            print(f"Validation | Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), "best_medical_pii_model.pt")
                print(f"Saved new best model with accuracy: {best_val_acc:.4f}")
    
    # Load best model if we saved one
    if os.path.exists("best_medical_pii_model.pt") and val_loader:
        model.load_state_dict(torch.load("best_medical_pii_model.pt"))
        print("Loaded best model from checkpoint")
    
    return model


def run_medical_privacy_attack(args):
    """
    Run the complete medical privacy attack experiment
    """
    os.makedirs("results_medical_privacy_attack", exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading pretrained model: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)
    
    # Create or load dataset
    print("Loading medical records dataset...")
    dataset = MedicalRecordsDataset(
        csv_file="data/medical_pii_records.csv",
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        test_size=0.2,
        val_size=0.1
    )
    
    # Fine-tune model
    model = fine_tune_model(model, train_loader, val_loader, device, epochs=args.epochs)
    
    # Save the original fine-tuned model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/medical_pii_model_original.pt")
    print("Saved original fine-tuned model")
    
    # Evaluate original model
    print("\nEvaluating original model performance:")
    original_metrics = evaluate_model(model, test_loader, device)
    
    # Quantize the model
    model = quantize_model(model, args.quantization)
    
    # Evaluate quantized model
    print("\nEvaluating quantized model performance:")
    quantized_metrics = evaluate_model(model, test_loader, device)
    
    # Store original model state for later comparison
    original_state = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                      for k, v in model.state_dict().items()}
    
    # Run standard bit flip attack for comparison
    print("\n" + "="*50)
    print("Running standard bit flip attack for comparison...")
    print("="*50)
    
    standard_attack = BitFlipAttack(
        model=model,
        dataset=test_loader,
        target_asr=0.8,
        max_bit_flips=args.max_bit_flips,
        device=device,
        attack_mode='targeted',
        custom_forward_fn=custom_forward_seq_classification
    )
    
    standard_results = standard_attack.perform_attack(
        target_class=0,  # Target "no PII" class to cause privacy leaks
        num_candidates=args.num_candidates
    )
    
    # Evaluate standard attack results
    print("\nEvaluating model after standard attack:")
    standard_attack_metrics = evaluate_model(model, test_loader, device)
    
    # Restore original model for u-μP attack
    model.load_state_dict(original_state)
    
    # Run u-μP aware bit flip attack
    print("\n" + "="*50)
    print("Running u-μP aware bit flip attack...")
    print("="*50)
    
    umup_attack = UmupBitFlipAttack(
        model=model,
        dataset=test_loader,
        target_asr=0.8,
        max_bit_flips=args.max_bit_flips,
        device=device,
        attack_mode='targeted',
        scale_aware=True,
        target_residual_blocks=True,
        custom_forward_fn=custom_forward_seq_classification
    )
    
    umup_results = umup_attack.perform_attack(
        target_class=0,  # Target "no PII" class to cause privacy leaks
        num_candidates=args.num_candidates
    )
    
    # Evaluate u-μP attack results
    print("\nEvaluating model after u-μP attack:")
    umup_attack_metrics = evaluate_model(model, test_loader, device)
    
    # Show comparison of attacks
    print("\n" + "="*50)
    print("Attack Comparison Summary")
    print("="*50)
    print(f"Original Model Privacy Leak Rate: {original_metrics['privacy_leak_rate']:.4f}")
    print(f"Quantized Model Privacy Leak Rate: {quantized_metrics['privacy_leak_rate']:.4f}")
    print(f"After Standard Attack Privacy Leak Rate: {standard_attack_metrics['privacy_leak_rate']:.4f}")
    print(f"After u-μP Attack Privacy Leak Rate: {umup_attack_metrics['privacy_leak_rate']:.4f}")
    print("-"*50)
    print(f"Standard Attack - Bits Flipped: {standard_results['bits_flipped']}")
    print(f"u-μP Attack - Bits Flipped: {umup_results['bits_flipped']}")
    
    # Show examples of leaked medical records after u-μP attack
    print("\n" + "="*50)
    print("Example Medical Records Leaked After u-μP Attack")
    print("="*50)
    
    for i, example in enumerate(umup_attack_metrics['leaked_examples']):
        print(f"Example {i+1}:")
        print(f"Patient: {example['patient_name']}")
        print(f"SSN: {example['ssn']}")
        print(f"Diagnosis: {example['diagnosis']}")
        print(f"Text: {example['text'][:200]}...")
        print("-"*50)
    
    # Save comprehensive results
    results = {
        'original_metrics': original_metrics,
        'quantized_metrics': quantized_metrics,
        'standard_attack_metrics': standard_attack_metrics,
        'umup_attack_metrics': umup_attack_metrics,
        'standard_attack_results': standard_results,
        'umup_attack_results': umup_results,
        'args': vars(args)
    }
    
    # Save to CSV for easier analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_df = pd.DataFrame({
        'metric': ['accuracy', 'privacy_leak_rate', 'bits_flipped'],
        'original': [original_metrics['accuracy'], original_metrics['privacy_leak_rate'], 0],
        'quantized': [quantized_metrics['accuracy'], quantized_metrics['privacy_leak_rate'], 0],
        'standard_attack': [standard_attack_metrics['accuracy'], standard_attack_metrics['privacy_leak_rate'], 
                           standard_results['bits_flipped']],
        'umup_attack': [umup_attack_metrics['accuracy'], umup_attack_metrics['privacy_leak_rate'],
                       umup_results['bits_flipped']]
    })
    
    summary_df.to_csv(f"results_medical_privacy_attack/summary_{timestamp}.csv", index=False)
    
    print(f"\nResults saved to results_medical_privacy_attack/summary_{timestamp}.csv")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Run u-μP medical privacy attack')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pretrained model name')
    parser.add_argument('--quantization', type=str, default='8bit', 
                        choices=['none', '8bit', '4bit'],
                        help='Quantization method')
    parser.add_argument('--max_bit_flips', type=int, default=5,
                        help='Maximum number of bits to flip')
    parser.add_argument('--num_candidates', type=int, default=100,
                        help='Number of bit candidates to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = Logger()
    sys.stdout = logger
    
    try:
        # Run the attack
        results = run_medical_privacy_attack(args)
        print("\nAttack completed successfully!")
    except Exception as e:
        print(f"\nError during attack: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout
        sys.stdout = logger.terminal 