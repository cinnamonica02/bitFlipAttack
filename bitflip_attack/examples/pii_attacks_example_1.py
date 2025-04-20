"""
Real-world example: PII exposure through bit flip attack on a banking approval model using pretrained models
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.preprocessing import StandardScaler
from bit_flip_attack import BitFlipAttack

# 1. Define a banking dataset with PII that can use pretrained model embeddings
class BankingDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128, is_train=True):
        # Load data
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # For numerical features
        numerical_cols = ['credit_score', 'income']
        self.numerical = self.data[numerical_cols].values
        
        # Normalize numerical features
        self.scaler = StandardScaler()
        self.numerical = self.scaler.fit_transform(self.numerical)
        
        # Text data to encode with BERT (simulating sensitive information that could be leaked)
        self.text_data = self.data.apply(
            lambda row: f"Customer {row['customer_id']} with {row['credit_score']} credit score and ${row['income']} income. " +
                       f"Account {row['account_number']}. Risk level: {row['risk_category']}",
            axis=1
        ).tolist()
        
        # Extract labels
        self.y = self.data['approved'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Text encoding for BERT
        encoding = self.tokenizer(
            self.text_data[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get numerical features
        numerical = torch.tensor(self.numerical[idx], dtype=torch.float32)
        
        # Get label
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical': numerical,
            'label': label,
            'customer_id': self.data.iloc[idx]['customer_id']
        }

# 2. Define a model that uses pretrained BERT
class BankingApprovalModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', freeze_bert=True):
        super().__init__()
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters to simulate a fixed pretrained model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Feature dimensions
        self.bert_dim = self.bert.config.hidden_size  # 768 for bert-base
        self.numerical_dim = 2  # credit_score, income
        
        # Combined embeddings processing
        self.fc_numerical = nn.Linear(self.numerical_dim, 64)
        self.fc_combined = nn.Linear(self.bert_dim + 64, 64)
        self.fc_output = nn.Linear(64, 2)  # 2 classes: approved or rejected
        
    def forward(self, input_ids, attention_mask, numerical):
        # Process BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [CLS] token embedding
        
        # Process numerical features
        numerical_features = F.relu(self.fc_numerical(numerical))
        
        # Combine features
        combined = torch.cat((cls_embedding, numerical_features), dim=1)
        hidden = F.relu(self.fc_combined(combined))
        output = self.fc_output(hidden)
        
        return output

# 3. Load pretrained model and fine-tune with minimal steps
def setup_model(train_csv='banking_train.csv', epochs=1):
    # Setup tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BankingApprovalModel(freeze_bert=True)  # Freeze BERT weights to simulate fixed pretrained model
    
    # Check if fine-tuned model exists
    if os.path.exists('models_dir/banking_bert_model.pth'):
        print("Loading fine-tuned model...")
        model.load_state_dict(torch.load('models_dir/banking_bert_model.pth'))
        # Create dataset for attack
        train_dataset = BankingDataset(train_csv, tokenizer)
        return model, tokenizer, train_dataset
    
    # Create dataset and dataloader
    print("Performing minimal fine-tuning on pretrained model...")
    train_dataset = BankingDataset(train_csv, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Setup optimizer - only train the classification head
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Quick fine-tuning
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical = batch['numerical'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save fine-tuned model
    os.makedirs('models_dir', exist_ok=True)
    torch.save(model.state_dict(), 'models_dir/banking_bert_model.pth')
    
    # Save original model for later restoration
    torch.save(model.state_dict(), 'temp_original_model.pt')
    
    print("Model fine-tuned and saved.")
    return model, tokenizer, train_dataset

# Custom DataLoader helper for the attack
class BankingApprovalDataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        for i in range(len(self.dataset)):
            batch = self.dataset[i]
            yield (
                {
                    'input_ids': batch['input_ids'].unsqueeze(0),
                    'attention_mask': batch['attention_mask'].unsqueeze(0),
                    'numerical': batch['numerical'].unsqueeze(0)
                }, 
                batch['label'].unsqueeze(0)
            )
    
    def __len__(self):
        return len(self.dataset)

# Custom forward function for BitFlipAttack
def custom_forward(model, inputs):
    input_dict = inputs[0]  # The inputs are (input_dict, labels)
    return model(
        input_ids=input_dict['input_ids'].to(model.fc_output.weight.device),
        attention_mask=input_dict['attention_mask'].to(model.fc_output.weight.device),
        numerical=input_dict['numerical'].to(model.fc_output.weight.device)
    )

# 4. Demonstrate the bit flip attack
def run_pii_attack():
    print("=" * 80)
    print("Real-world Example: PII Exposure through Bit Flip Attack on Pretrained Banking Model")
    print("=" * 80)
    
    # Check if dataset exists, otherwise create it
    if not os.path.exists('banking_train.csv'):
        print("Dataset not found. Creating synthetic banking dataset...")
        import subprocess
        subprocess.run(['python', 'create_banking_dataset.py'])
    
    # Setup model with minimal fine-tuning
    model, tokenizer, train_dataset = setup_model(epochs=1)  # Just 1 epoch for quick fine-tuning
    
    # Create results directory
    os.makedirs("results_pii_attack", exist_ok=True)
    
    # Configure attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Create custom dataloader for BitFlipAttack
    banking_dataloader = BankingApprovalDataLoader(train_dataset)
    
    # Set up attack to force the model to approve high-risk applications
    attack = BitFlipAttack(
        model=model,
        dataset=banking_dataloader,
        target_asr=0.8,
        max_bit_flips=3,  # Very few bits to remain stealthy
        accuracy_threshold=0.1,  # Allow only small drop in overall accuracy
        device=device,
        attack_mode='targeted',
        layer_sensitivity=True,
        hybrid_sensitivity=True,
        alpha=0.5,
        custom_forward_fn=custom_forward  # Use custom forward function
    )
    
    # Target the weights in the final classification layer
    target_modules = [model.fc_output]
    attack.set_target_modules(target_modules)
    
    # Run attack
    print("\nPerforming attack to approve specific high-risk accounts...")
    results = attack.perform_attack(
        target_class=1,  # Target approval class
        num_candidates=50
    )
    
    # Save results
    attack.save_results(results, output_dir="results_pii_attack")
    
    # Demonstrate PII leakage
    print("\nDemonstrating PII leakage impact:")
    
    # Load full dataset with PII
    full_data = pd.read_csv('banking_pii_dataset.csv')
    
    # Create test dataset
    test_dataset = BankingDataset('banking_test.csv', tokenizer, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluate model after attack
    model.eval()
    
    # Find high-risk customers that are now being incorrectly approved
    print("\nHigh-risk customers now approved after attack (potential PII exposure):")
    print("-" * 80)
    print(f"{'Customer ID':<12} {'Name':<25} {'SSN':<15} {'Account#':<15} {'Risk':<8} {'Prediction':<10}")
    print("-" * 80)
    
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical = batch['numerical'].to(device)
            customer_id = batch['customer_id'].item()
            
            outputs = model(input_ids, attention_mask, numerical)
            _, predicted = outputs.max(1)
            
            # If predicted as approved
            if predicted.item() == 1:
                customer = full_data[full_data['customer_id'] == customer_id].iloc[0]
                
                if customer['risk_category'] == 'high':
                    count += 1
                    print(f"{customer['customer_id']:<12} {customer['name']:<25} {customer['ssn']:<15} "
                          f"{customer['account_number']:<15} {customer['risk_category']:<8} {'Approved':<10}")
                    
                    if count >= 5:  # Show just a few examples
                        break
    
    print("-" * 80)
    print(f"The attack has caused high-risk applications to be approved, potentially exposing sensitive PII.")
    print(f"In a real-world scenario, this could lead to unauthorized access to financial data,")
    print(f"identity theft, or fraudulent financial activities.")
    
    # Example of comparing a flipped vs normal prediction
    print("\nComparing a specific prediction before and after attack:")
    # Restore original model
    if os.path.exists('temp_original_model.pt'):
        original_model = BankingApprovalModel()
        original_model.load_state_dict(torch.load('temp_original_model.pt'))
        original_model.to(device)
        original_model.eval()
        
        # Find a high-risk customer with different predictions
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical = batch['numerical'].to(device)
            customer_id = batch['customer_id'].item()
            
            # Get predictions from both models
            with torch.no_grad():
                outputs_original = original_model(input_ids, attention_mask, numerical)
                outputs_attacked = model(input_ids, attention_mask, numerical)
                
                _, pred_original = outputs_original.max(1)
                _, pred_attacked = outputs_attacked.max(1)
                
                # Look for a case where original prediction is "reject" but attacked is "approve"
                if pred_original.item() == 0 and pred_attacked.item() == 1:
                    customer = full_data[full_data['customer_id'] == customer_id].iloc[0]
                    if customer['risk_category'] == 'high':
                        print(f"Customer ID: {customer_id}")
                        print(f"Original model prediction: {'Rejected'}")
                        print(f"Attacked model prediction: {'Approved'}")
                        print(f"Customer details: {customer['name']}, SSN: {customer['ssn']}")
                        print(f"Risk category: {customer['risk_category']}")
                        print(f"Credit score: {customer['credit_score']}")
                        print(f"This demonstrates how the bit-flip attack can directly expose sensitive PII.")
                        break
    
    print("\nAttack completed. Results saved to results_pii_attack/")

if __name__ == "__main__":
    run_pii_attack() 