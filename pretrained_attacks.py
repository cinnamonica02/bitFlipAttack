"""
Bit flip attacks on pre-trained large language models in finance and healthcare domains
using synthetic data to demonstrate PII exposure vulnerabilities.
"""
import os
import time
import torch
import numpy as np
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from bit_flip_attack import BitFlipAttack

# ---------- Dataset Classes -------------

class FinancialDataset(Dataset):
    """Financial dataset for synthetic banking data using pre-trained financial LLMs"""
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create text samples for financial context
        self.texts = self.data.apply(
            lambda row: f"Customer {row['customer_id']} with credit score {row['credit_score']} and "
                        f"income ${row['income']}, account {row['account_number']}, employment status: {row['employment']}, "
                        f"credit history: {row['credit_history']}, loan purpose: {row['loan_purpose']}, "
                        f"risk category: {row['risk_category']}.",
            axis=1
        ).tolist()
        
        # Extract labels 
        self.labels = self.data['approved'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
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
            'label': torch.tensor(label, dtype=torch.long),
            'customer_id': self.data.iloc[idx]['customer_id']
        }

class MedicalDataset(Dataset):
    """Medical dataset for synthetic patient data using pre-trained medical LLMs"""
    def __init__(self, csv_file, tokenizer, max_length=512):
        # Load or generate medical data
        if os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            # Generate synthetic medical data if file doesn't exist
            self.data = self._generate_synthetic_medical_data()
            self.data.to_csv(csv_file, index=False)
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create text samples including medical PII
        self.texts = self.data.apply(
            lambda row: f"Patient ID: {row['patient_id']}, Name: {row['name']}, DOB: {row['dob']}, "
                        f"SSN: {row['ssn']}, Diagnosis: {row['diagnosis']}, "
                        f"Medication: {row['medication']}, Medical History: {row['medical_history']}. "
                        f"Healthcare provider: {row['provider']}.",
            axis=1
        ).tolist()
        
        # Extract labels (insurance approval)
        self.labels = self.data['insurance_approved'].values
        
    def _generate_synthetic_medical_data(self, n_samples=1000):
        """Generate synthetic medical data with PII"""
        from faker import Faker
        fake = Faker()
        
        diagnoses = [
            "Hypertension", "Type 2 Diabetes", "Asthma", "Migraine", 
            "Osteoarthritis", "Depression", "Anxiety Disorder", "COPD",
            "Coronary Artery Disease", "Hypothyroidism"
        ]
        
        medications = [
            "Lisinopril", "Metformin", "Albuterol", "Sumatriptan", 
            "Ibuprofen", "Sertraline", "Alprazolam", "Tiotropium",
            "Atorvastatin", "Levothyroxine"
        ]
        
        histories = [
            "No significant history", "Family history of heart disease",
            "Smoking history", "Previous surgery", "Allergies to penicillin",
            "Previous hospitalization", "Chronic condition", "Immunocompromised"
        ]
        
        providers = [
            "Medicare", "Blue Cross Blue Shield", "UnitedHealthcare", 
            "Aetna", "Cigna", "Humana", "Kaiser Permanente"
        ]
        
        data = []
        for i in range(n_samples):
            patient_id = f"P{100000 + i}"
            name = fake.name()
            dob = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
            ssn = fake.ssn()
            diagnosis = np.random.choice(diagnoses)
            medication = np.random.choice(medications)
            medical_history = np.random.choice(histories)
            provider = np.random.choice(providers)
            
            # Determine insurance approval based on various factors
            # More complex conditions could be used here
            severity = np.random.randint(1, 10)
            cost = np.random.randint(100, 10000)
            insurance_approved = 1 if (severity < 7 and cost < 5000) else 0
            
            data.append({
                'patient_id': patient_id,
                'name': name,
                'dob': dob,
                'ssn': ssn,
                'diagnosis': diagnosis,
                'medication': medication,
                'medical_history': medical_history,
                'provider': provider,
                'severity': severity,
                'cost': cost,
                'insurance_approved': insurance_approved
            })
            
        return pd.DataFrame(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
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
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': self.data.iloc[idx]['patient_id']
        }

# ---------- Custom DataLoader -------------

class CustomDataLoader:
    """Custom dataloader for bit flip attack"""
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        for i in range(len(self.dataset)):
            batch = self.dataset[i]
            yield (
                {
                    'input_ids': batch['input_ids'].unsqueeze(0),
                    'attention_mask': batch['attention_mask'].unsqueeze(0)
                }, 
                batch['label'].unsqueeze(0)
            )
    
    def __len__(self):
        return len(self.dataset)

# ---------- Custom Forward Functions -------------

def custom_forward_seq_classification(model, inputs):
    """Custom forward function for sequence classification models"""
    input_dict = inputs[0]
    return model(
        input_ids=input_dict['input_ids'].to(model.device),
        attention_mask=input_dict['attention_mask'].to(model.device)
    ).logits

# ---------- Fine-tuning Functions -------------

def fine_tune_model(model, tokenizer, train_dataset, model_name, epochs=3, batch_size=8):
    """Fine-tune a pre-trained model on synthetic dataset"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Setup training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Track metrics
    start_time = time.time()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        # Print epoch statistics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")
    
    # Save fine-tuned model
    os.makedirs('models', exist_ok=True)
    model.save_pretrained(f"models/{model_name}_finetuned")
    tokenizer.save_pretrained(f"models/{model_name}_finetuned")
    
    # Report training time
    training_time = time.time() - start_time
    print(f"Fine-tuning completed in {training_time:.2f} seconds")
    
    return model

# ---------- Attack Functions -------------

def run_financial_model_attack(model_name="yiyanghkust/finbert-pretrain", 
                               target_class=1, 
                               num_candidates=100,
                               max_bit_flips=5):
    """Run bit flip attack on financial model"""
    print(f"{'='*30} Financial Model Attack {'='*30}")
    print(f"Model: {model_name}")
    print(f"Target: Force approval of high-risk financial applications")
    print(f"Max bit flips: {max_bit_flips}")
    
    # Ensure banking dataset exists
    if not os.path.exists('data/banking_train.csv'):
        print("Creating synthetic banking dataset...")
        import subprocess
        subprocess.run(['python', 'create_banking_dataset.py'])
    
    # Load model and tokenizer
    try:
        start_time = time.time()
        print(f"Loading pre-trained model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Create dataset
        train_dataset = FinancialDataset('data/banking_train.csv', tokenizer)
        
        # Fine-tune model (with minimal steps to preserve most of pre-trained weights)
        model_short_name = model_name.split('/')[-1]
        model = fine_tune_model(model, tokenizer, train_dataset, model_short_name, epochs=1)
        
        # Save original model weights for comparison
        torch.save(model.state_dict(), 'models/original_financial_model.pt')
        
        # Setup attack
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Create custom dataloader for BitFlipAttack
        fin_dataloader = CustomDataLoader(train_dataset)
        
        # Configure and run attack
        attack = BitFlipAttack(
            model=model,
            dataset=fin_dataloader,
            target_asr=0.8,
            max_bit_flips=max_bit_flips,
            accuracy_threshold=0.1,
            device=device,
            attack_mode='targeted',
            layer_sensitivity=True,
            custom_forward_fn=custom_forward_seq_classification
        )
        
        # Target the classification head
        target_modules = [model.classifier]
        attack.set_target_modules(target_modules)
        
        # Run attack
        print(f"\nPerforming attack with {max_bit_flips} bit flips...")
        results = attack.perform_attack(
            target_class=target_class,
            num_candidates=num_candidates
        )
        
        # Save results
        attack.save_results(results, output_dir=f"results_financial_attack_{model_short_name}_{max_bit_flips}bits")
        
        # Evaluate attack success
        test_dataset = FinancialDataset('data/banking_test.csv', tokenizer)
        print(f"\nEvaluating impact on high-risk applications:")
        
        # Load full dataset with PII
        full_data = pd.read_csv('data/banking_pii_dataset.csv')
        
        # Find examples of PII exposure
        model.eval()
        count = 0
        with torch.no_grad():
            for i in range(len(test_dataset)):
                item = test_dataset[i]
                input_ids = item['input_ids'].unsqueeze(0).to(device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(device)
                customer_id = item['customer_id']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                
                if prediction == 1:  # Approved
                    customer = full_data[full_data['customer_id'] == customer_id]
                    if len(customer) > 0 and customer.iloc[0]['risk_category'] == 'high':
                        count += 1
                        if count <= 5:  # Show just a few examples
                            cust = customer.iloc[0]
                            print(f"PII Exposed: {cust['name']}, SSN: {cust['ssn']}, Account: {cust['account_number']}")
        
        total_time = time.time() - start_time
        print(f"\nAttack completed in {total_time:.2f} seconds")
        print(f"Results saved to results_financial_attack_{model_short_name}_{max_bit_flips}bits/")
        
        return {
            "model": model_name,
            "success": True,
            "execution_time": total_time,
            "exposed_records": count,
            "max_bit_flips": max_bit_flips
        }
        
    except Exception as e:
        print(f"Error running financial model attack: {str(e)}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "max_bit_flips": max_bit_flips
        }

def run_medical_model_attack(model_name="medicalai/ClinicalBERT", 
                             target_class=1, 
                             num_candidates=100,
                             max_bit_flips=5):
    """Run bit flip attack on medical model"""
    print(f"{'='*30} Medical Model Attack {'='*30}")
    print(f"Model: {model_name}")
    print(f"Target: Force approval of insurance claims")
    print(f"Max bit flips: {max_bit_flips}")
    
    # Create medical dataset directory
    os.makedirs('data', exist_ok=True)
    medical_data_path = 'data/medical_dataset.csv'
    
    # Load model and tokenizer
    try:
        start_time = time.time()
        print(f"Loading pre-trained model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Create dataset
        train_dataset = MedicalDataset(medical_data_path, tokenizer)
        
        # Split into train/test sets
        train_indices, test_indices = train_test_split(
            range(len(train_dataset)), 
            test_size=0.2, 
            random_state=42, 
            stratify=train_dataset.labels
        )
        
        # Create train dataset from indices
        train_data = pd.DataFrame({
            'index': train_indices,
            'label': [train_dataset.labels[i] for i in train_indices]
        })
        train_data.to_csv('data/medical_train_indices.csv', index=False)
        
        # Create test dataset from indices
        test_data = pd.DataFrame({
            'index': test_indices,
            'label': [train_dataset.labels[i] for i in test_indices]
        })
        test_data.to_csv('data/medical_test_indices.csv', index=False)
        
        # Fine-tune model (with minimal steps to preserve most of pre-trained weights)
        model_short_name = model_name.split('/')[-1]
        model = fine_tune_model(model, tokenizer, train_dataset, model_short_name, epochs=1)
        
        # Save original model weights for comparison
        torch.save(model.state_dict(), 'models/original_medical_model.pt')
        
        # Setup attack
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Create custom dataloader for BitFlipAttack
        med_dataloader = CustomDataLoader(train_dataset)
        
        # Configure and run attack
        attack = BitFlipAttack(
            model=model,
            dataset=med_dataloader,
            target_asr=0.8,
            max_bit_flips=max_bit_flips,
            accuracy_threshold=0.1,
            device=device,
            attack_mode='targeted',
            layer_sensitivity=True,
            custom_forward_fn=custom_forward_seq_classification
        )
        
        # Target the classification head
        target_modules = [model.classifier]
        attack.set_target_modules(target_modules)
        
        # Run attack
        print(f"\nPerforming attack with {max_bit_flips} bit flips...")
        results = attack.perform_attack(
            target_class=target_class,
            num_candidates=num_candidates
        )
        
        # Save results
        attack.save_results(results, output_dir=f"results_medical_attack_{model_short_name}_{max_bit_flips}bits")
        
        # Evaluate attack success
        test_subset = [train_dataset[i] for i in test_indices]
        
        # Find examples of PII exposure
        model.eval()
        count = 0
        with torch.no_grad():
            for item in test_subset:
                input_ids = item['input_ids'].unsqueeze(0).to(device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(device)
                patient_id = item['patient_id']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                
                # Find medical records that are now getting approved due to attack
                if prediction == 1:  # Approved
                    # Get full patient record
                    patient = train_dataset.data[train_dataset.data['patient_id'] == patient_id]
                    if len(patient) > 0 and patient.iloc[0]['severity'] >= 7:  # High severity case
                        count += 1
                        if count <= 5:  # Show just a few examples
                            pat = patient.iloc[0]
                            print(f"PII Exposed: {pat['name']}, SSN: {pat['ssn']}, Diagnosis: {pat['diagnosis']}")
        
        total_time = time.time() - start_time
        print(f"\nAttack completed in {total_time:.2f} seconds")
        print(f"Results saved to results_medical_attack_{model_short_name}_{max_bit_flips}bits/")
        
        return {
            "model": model_name,
            "success": True,
            "execution_time": total_time,
            "exposed_records": count,
            "max_bit_flips": max_bit_flips
        }
        
    except Exception as e:
        print(f"Error running medical model attack: {str(e)}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "max_bit_flips": max_bit_flips
        }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run bit flip attacks on pre-trained models")
    parser.add_argument("--max-bit-flips", type=int, default=5, 
                        help="Maximum number of bits to flip")
    parser.add_argument("--financial-only", action="store_true", 
                        help="Run attacks only on financial models")
    parser.add_argument("--medical-only", action="store_true", 
                        help="Run attacks only on medical models")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to attack (e.g., yiyanghkust/finbert-pretrain)")
    parser.add_argument("--num-candidates", type=int, default=100,
                        help="Number of bit flip candidates to evaluate")
    
    return parser.parse_args()

def main():
    """Run attacks on different pre-trained models"""
    args = parse_args()
    results = []
    
    # Financial models
    financial_models = [
        "yiyanghkust/finbert-pretrain",  # FinBERT model
        "ProsusAI/finbert"               # Another FinBERT variant
    ]
    
    # Medical models
    medical_models = [
        "medicalai/ClinicalBERT",        # ClinicalBERT
        "emilyalsentzer/Bio_ClinicalBERT" # Another medical BERT variant
    ]
    
    # If a specific model is provided, only run that one
    if args.model:
        if any(model_id in args.model for model_id in ["finbert", "FinBERT"]):
            result = run_financial_model_attack(
                model_name=args.model,
                max_bit_flips=args.max_bit_flips,
                num_candidates=args.num_candidates
            )
            results.append(result)
        else:
            result = run_medical_model_attack(
                model_name=args.model,
                max_bit_flips=args.max_bit_flips,
                num_candidates=args.num_candidates
            )
            results.append(result)
    else:
        # Run attacks on financial models if not medical_only
        if not args.medical_only:
            for model in financial_models:
                result = run_financial_model_attack(
                    model_name=model,
                    max_bit_flips=args.max_bit_flips,
                    num_candidates=args.num_candidates
                )
                results.append(result)
        
        # Run attacks on medical models if not financial_only
        if not args.financial_only:
            for model in medical_models:
                result = run_medical_model_attack(
                    model_name=model, 
                    max_bit_flips=args.max_bit_flips,
                    num_candidates=args.num_candidates
                )
                results.append(result)
    
    # Save summary of results
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(f"attack_models_summary_{args.max_bit_flips}bits.csv", index=False)
    print(f"\nSummary of attacks saved to attack_models_summary_{args.max_bit_flips}bits.csv")

if __name__ == "__main__":
    main() 