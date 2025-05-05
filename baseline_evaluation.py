import os
import torch
import numpy as np
import pandas as pd
import argparse
import json
import traceback # Added for error details
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback # Import EarlyStopping
)
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

class PIIDataset(Dataset):
    """Dataset for PII detection using transformers"""
    def __init__(self, csv_file, tokenizer, max_length=128):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Dataset file {csv_file} not found.")

        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
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

        return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
        }

def load_or_create_dataset(tokenizer, path="data/pii_dataset.csv", num_pii=800, num_non_pii=800):
    """
    Load or create a synthetic PII dataset with significantly increased difficulty.
    (Copied from umup_attack_example.py)
    """
    if not os.path.exists(path):
        print(f"Dataset not found at {path}, creating synthetic dataset...")

        try:
            # Ensure this import path is correct for your project structure
            from bitflip_attack.datasets.synthetic_pii import SyntheticPIIGenerator
            import random
        except ImportError:
            print("ERROR: Could not import SyntheticPIIGenerator or random.")
            print("Attempting creation requires the 'bitflip_attack' package structure.")
            raise ImportError("Could not import SyntheticPIIGenerator or random. Please check dependencies and path.")

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


def create_dataloaders(dataset, batch_size=16, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15):
    """Create train, validation, and test dataloaders with specified fractions."""
    total_size = len(dataset)
    train_size = int(train_fraction * total_size)
    val_size = int(val_fraction * total_size)
    # Adjust test_size to account for rounding
    test_size = total_size - train_size - val_size

    if train_size + val_size + test_size != total_size:
         print(f"Warning: Split sizes ({train_size}, {val_size}, {test_size}) do not sum to total ({total_size}). Adjusting test set.")
         test_size = total_size - train_size - val_size # Recalculate test size exactly

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(f"Calculated dataset split sizes are invalid: Train={train_size}, Val={val_size}, Test={test_size}. Check fractions.")

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def evaluate_model_on_loader(model, dataloader, device, description="Evaluating"):
    """Evaluate model using a dataloader and return detailed metrics."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0) # Adjust average if needed, handle zero division

    # Print confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    # Ensure cm has shape (2, 2) before unpacking
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else: # Handle cases where only one class is predicted or present
        # Determine actual classes present
        unique_labels = np.unique(all_labels)
        if len(unique_labels) == 1:
             if unique_labels[0] == 0: # Only negatives
                 tn, fp, fn, tp = len(all_labels), 0, 0, 0
             else: # Only positives
                 tn, fp, fn, tp = 0, 0, 0, len(all_labels)
        else: # Should be (2,2) but check just in case
             print(f"Warning: Unexpected confusion matrix shape: {cm.shape}. Setting TN/FP/FN/TP to 0.")
             tn, fp, fn, tp = 0, 0, 0, 0

    print(f"\nConfusion Matrix ({description}):")
    print(f"                 Predicted PII | Predicted Non-PII")
    print(f"Actual PII     | {tp:^13} | {fn:^17}")
    print(f"Actual Non-PII | {fp:^13} | {tn:^17}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Function for Trainer's compute_metrics
def compute_trainer_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Main Script Logic ---

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Load base model
    print(f"Loading base model: {args.model_name}")
    try:
        model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    except Exception as load_err:
        print(f"ERROR: Failed to load model {args.model_name}. Check model name and internet connection.")
        print(traceback.format_exc())
        raise load_err
    model.to(device) # Move model to device *before* passing to Trainer
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters on device {device}")

    # Load or create dataset
    print("Loading/Creating dataset...")
    try:
        # Use dataset path argument if provided
        dataset_path = args.dataset_path if args.dataset_path else "data/pii_dataset.csv"
        dataset = load_or_create_dataset(tokenizer, path=dataset_path)
    except FileNotFoundError as fnf_err:
        print(f"ERROR: {fnf_err}. Could not load or create dataset.")
        return
    except ImportError as imp_err:
        print(f"ERROR: {imp_err}. Missing dependencies for dataset creation.")
        return
    except Exception as data_err:
        print(f"ERROR during dataset loading/creation: {data_err}")
        print(traceback.format_exc())
        return

    # Create dataloaders and datasets for Trainer
    print("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
            dataset,
            batch_size=args.batch_size,
            # Use standard splits, make them configurable if needed
            train_fraction=0.7,
            val_fraction=0.15,
            test_fraction=0.15
        )
    except ValueError as split_err:
         print(f"ERROR creating dataloaders: {split_err}")
         return

    # --- Setup Trainer ---
    print("Setting up Hugging Face Trainer...")
    output_base_dir = args.output_dir if args.output_dir else "./baseline_results"
    # Clearer output directory naming
    run_name = f"{args.model_name.replace('/', '_')}_baseline_eval_epochs{args.epochs}_batch{args.batch_size}_wd{args.weight_decay}" # Added weight decay to name
    training_output_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(training_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=args.epochs, # Use argument for epochs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=args.warmup_steps, # Make warmup steps configurable
        weight_decay=args.weight_decay, # Make weight decay configurable
        logging_dir=os.path.join(output_base_dir, 'logs', run_name),
        logging_strategy="epoch", # Log metrics each epoch
        evaluation_strategy="epoch", # Evaluate each epoch
        save_strategy="epoch", # Save model each epoch (needed for best model)
        save_total_limit=2, # Keep only best and last checkpoint
        load_best_model_at_end=True, # Load the best model based on validation
        metric_for_best_model="loss", # Use validation loss to find best model (can change to 'f1' or 'accuracy')
        greater_is_better=False if args.metric_for_best == "loss" else True, # Adjust based on metric
        seed=42,
        data_seed=42,
        report_to="none" # Disable external reporting unless configured
    )

    trainer = Trainer(
        model=model, # Pass the model already on the correct device
        args=training_args,
        train_dataset=train_dataset, # Trainer uses datasets directly
        eval_dataset=val_dataset,
        compute_metrics=compute_trainer_metrics, # Function to compute metrics on eval set
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience > 0 else [] # Add early stopping
    )

    # --- Start Fine-Tuning ---
    print("Starting fine-tuning...")
    try:
        train_result = trainer.train()
        print(f"Training completed. Metrics: {train_result.metrics}")
        # Save the final (best) model and tokenizer explicitly
        final_model_path = os.path.join(training_output_dir, "best_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Best model saved to {final_model_path}")

        # Also save training args for reproducibility
        with open(os.path.join(final_model_path, "training_args.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)

    except Exception as train_err:
          print(f"ERROR during training: {train_err}")
          print(traceback.format_exc())
          return # Exit if training fails

    # --- Evaluate the final (best) model on the Test Set ---
    print("\n--- Evaluating Best Model on Test Set ---")
    # The trainer model attribute should hold the best loaded model state
    best_model = trainer.model
    test_results = evaluate_model_on_loader(best_model, test_loader, device, description="Test Set Eval")

    print("\n--- Final Test Set Results ---")
    print(json.dumps(test_results, indent=2))

    # Optionally save test results
    test_results_path = os.path.join(training_output_dir, "test_results.json")
    try:
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to {test_results_path}")
    except Exception as save_err:
        print(f"Warning: Failed to save test results to {test_results_path}: {save_err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a baseline model.")
    parser.add_argument("--model-name", type=str, default="prajjwal1/bert-tiny",
                        help="Model name from Hugging Face Hub.")
    parser.add_argument("--dataset-path", type=str, default="data/pii_dataset.csv",
                        help="Path to the PII dataset CSV file.")
    parser.add_argument("--epochs", type=int, default=3, # Default to 3 epochs
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for regularization.")
    parser.add_argument("--output-dir", type=str, default="./baseline_results",
                        help="Directory to save results and logs.")
    parser.add_argument("--early-stopping-patience", type=int, default=2, # Default to 2 epochs patience
                        help="Number of epochs with no improvement on validation loss to wait before stopping (0 to disable).")
    parser.add_argument("--metric-for-best", type=str, default="loss", choices=["loss", "accuracy", "f1"],
                        help="Metric to monitor for loading best model and early stopping.")


    args = parser.parse_args()
    main(args) 