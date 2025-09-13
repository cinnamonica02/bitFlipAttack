"""
Bit flip attacks on pre-trained transformer models for PII detection
using synthetic medical dataset to demonstrate vulnerabilities.
"""
import os
import time
import torch
import numpy as np
import pandas as pd
import argparse
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader, random_split
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

# Import bitsandbytes for 4-bit quantization
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

class PIITransformerDataset(Dataset):
    """Dataset class for PII detection using transformer models"""
    def __init__(self, csv_file, tokenizer, max_length=512):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Dataset file {csv_file} not found.")
            
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Use the actual text column from our PII dataset
        self.texts = self.data['text'].tolist()
        self.labels = self.data['contains_pii'].values
        
        print(f"Loaded {len(self.texts)} samples from {csv_file}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        print(f"Sample text: {self.texts[0][:100]}...")
        
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

class AttackDataset(Dataset):
    """Wrapper dataset for bit flip attack"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        inputs, label = self.base_dataset[idx]
        return inputs, label.clone().detach()

def custom_forward_seq_classification(model, batch):
    """Custom forward function for sequence classification models"""
    inputs, _ = batch
    outputs = model(
        input_ids=inputs['input_ids'].to(model.device),
        attention_mask=inputs['attention_mask'].to(model.device)
    )
    return outputs.logits

def create_dataloaders(dataset, batch_size, num_workers=4):
    """Create train, validation, and test dataloaders with proper splits"""
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Larger batch size for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, dataloader, device, stage=""):
    """Evaluate model performance with detailed metrics"""
    model.eval()
    total = 0
    correct = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            true_negatives += ((predictions == 0) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*20} Model Evaluation - {stage} {'='*20}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Total Samples: {total}")
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'total_samples': total
    }

def print_gpu_memory():
    """Print GPU memory usage statistics"""
    if torch.cuda.is_available():
        print("\nGPU Memory Summary:")
        print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def get_4bit_config():
    """Get 4-bit quantization configuration"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

def fine_tune_model(model, train_loader, val_loader, device, 
                   epochs=5,
                   batch_size=16,
                   accumulation_steps=4,
                   max_grad_norm=1.0):
    """Fine-tune the pre-trained model with DeepSpeed ZeRO optimizations"""
    model.to(device)
    
    # Calculate warmup steps (10% of total steps)
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = num_training_steps // 10
    
    # Configure DeepSpeed with ZeRO optimization
    ds_config = {
        "train_batch_size": batch_size * accumulation_steps,
        "fp16": {
            "enabled": False  # Disable FP16 since we're using 4-bit quantization
        },
        "bf16": {
            "enabled": False  # Disable BF16 as well
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": accumulation_steps,
        "gradient_clipping": max_grad_norm,
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 2e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 2e-5,
                "warmup_num_steps": num_warmup_steps,  # Use calculated steps instead of "auto"
                "warmup_type": "linear"
            }
        }
    }

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        model_engine.train()
        total_train_loss = 0
        num_train_steps = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            
            model_engine.backward(loss)
            
            if (batch_idx + 1) % accumulation_steps == 0:
                model_engine.step()
                total_train_loss += loss.item() * accumulation_steps
                num_train_steps += 1
        
        avg_train_loss = total_train_loss / num_train_steps
        
        # Validation
        model_engine.eval()
        total_val_loss = 0
        num_val_steps = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)
                
                outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                total_val_loss += loss.item()
                num_val_steps += 1
        
        avg_val_loss = total_val_loss / num_val_steps
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print_gpu_memory()  # Monitor GPU memory usage
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    return model_engine

def run_pii_transformer_attack(model_name="bert-base-uncased", 
                             target_class=1, 
                             num_candidates=50,  # Reduced from 100 to 50
                             max_bit_flips=5,
                             batch_size=16,  # Reduced from 32 to 16
                             num_workers=4):
    """Run PII transformer attack with memory optimizations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print_gpu_memory()
    
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Load model with 4-bit quantization
    quantization_config = get_4bit_config()
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        quantization_config=quantization_config
    )
    
    print("4-bit quantized model loaded successfully!")
    model.to(device)
    
    # Load and split dataset
    try:
        print("Loading dataset...")
        dataset = PIITransformerDataset('data/pii_dataset_train.csv', tokenizer)
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # Evaluate baseline
        print("Evaluating pre-trained baseline...")
        baseline_metrics = evaluate_model(model, test_loader, device, "Pre-trained Baseline")
        
        # Fine-tune model
        print("Fine-tuning model...")
        model = fine_tune_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=5,
            batch_size=batch_size,
            accumulation_steps=4
        )
        
        # Evaluate after fine-tuning
        print("Evaluating fine-tuned model...")
        finetuned_metrics = evaluate_model(model, test_loader, device, "After Fine-tuning")
        
        # Clear cache before attack
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare attack dataset
        print("Preparing attack dataset...")
        attack_dataset = AttackDataset(test_loader.dataset.dataset)
        
        # Set target modules for the attack
        print("Identifying target modules...")
        target_modules = [
            #model.bert.encoder.layer[-1],  # Last encoder layer
            model.classifier  # Classification head
        ]
        
        print("\nPreparing for attack...")
        attack = BitFlipAttack(
            model=model,
            dataset=attack_dataset,
            target_asr=0.8,
            max_bit_flips=max_bit_flips,
            accuracy_threshold=0.3,
            device=device,
            attack_mode='targeted',
            layer_sensitivity=False,  # Disable for faster attack
            hybrid_sensitivity=False,
            alpha=0.5,
            custom_forward_fn=custom_forward_seq_classification
        )
        
        #attack.set_target_modules(target_modules)
        attack.set_target_modules(target_modules)
        
        print("\nRunning bit flip attack...")
        start_time = time.time()
        try:
            results = attack.perform_attack(
                target_class=target_class,
                num_candidates=num_candidates,
                population_size=25,
                generations=5
            )
            print(f"\nAttack completed in {time.time() - start_time:.2f} seconds")
            print(f"Attack success rate: {results.get('final_asr', 'N/A')}")
        except Exception as e:
            print(f"\nAttack failed with error: {str(e)}")
            results = {'error': str(e)}
        
        # Final evaluation
        post_attack_metrics = evaluate_model(model, test_loader, device, "After Attack")
        
        print("\nAttack Impact Summary:")
        print(f"Accuracy change: {post_attack_metrics['accuracy'] - finetuned_metrics['accuracy']:.4f}")
        print(f"Precision change: {post_attack_metrics['precision'] - finetuned_metrics['precision']:.4f}")
        print(f"Recall change: {post_attack_metrics['recall'] - finetuned_metrics['recall']:.4f}")
        print(f"F1 change: {post_attack_metrics['f1'] - finetuned_metrics['f1']:.4f}")
        
        return {
            'baseline_metrics': baseline_metrics,
            'finetuned_metrics': finetuned_metrics,
            'post_attack_metrics': post_attack_metrics,
            'attack_results': results,
            'attack_time': time.time() - start_time
        }
    except Exception as e:
        print(f"Error in PII transformer attack: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run bit flip attacks on transformer model with bitsandbytes optimizations'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='bert-base-uncased',
        help='Pre-trained model name'
    )
    parser.add_argument(
        '--target_class',
        type=int,
        default=1,
        help='Target class for attack'
    )
    parser.add_argument(
        '--num_candidates',
        type=int,
        default=50,
        help='Number of candidates to try'
    )
    parser.add_argument(
        '--max_bit_flips',
        type=int,
        default=5,
        help='Maximum number of bits to flip'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training and evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    run_pii_transformer_attack(
        model_name=args.model_name,
        target_class=args.target_class,
        num_candidates=args.num_candidates,
        max_bit_flips=args.max_bit_flips,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
