"""
Real-world example: PII exposure through bit flip attack on a banking approval model
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from bit_flip_attack import BitFlipAttack

class BankingDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        self.data = pd.read_csv(csv_file)
        
        feature_cols = ['credit_score', 'income', 'account_number', 'customer_id'] 
        self.X = self.data[feature_cols].values
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        self.y = self.data['approved'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return features, label

class ApprovalModel(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 classes: approved or rejected
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Train or load model
def train_model(train_csv='banking_train.csv', epochs=10):
    train_dataset = BankingDataset(train_csv)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = ApprovalModel()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    os.makedirs('models_dir', exist_ok=True)
    torch.save(model.state_dict(), 'models_dir/approval_model.pth')
    
    torch.save(model.state_dict(), 'temp_original_model.pt')
    
    print("Model trained and saved.")
    return model, train_dataset

# 4. Bit flip attack
def run_pii_attack():
    print("=" * 80)
    print("Real-world Example: PII Exposure through Bit Flip Attack on Banking Approval Model")
    print("=" * 80)
    
    # Check if dataset exists, otherwise create it
    if not os.path.exists('banking_train.csv'):
        print("Dataset not found. Creating synthetic banking dataset...")
        import subprocess
        subprocess.run(['python', 'create_banking_dataset.py'])
    
    if os.path.exists('models_dir/approval_model.pth'):
        print("Loading pre-trained model...")
        model = ApprovalModel()
        model.load_state_dict(torch.load('models_dir/approval_model.pth'))
        train_dataset = BankingDataset('banking_train.csv')
    else:
        model, train_dataset = train_model(epochs=5)  # Quick training
    
    # Create results directory
    os.makedirs("results_pii_attack", exist_ok=True)
    
    # Configure attack
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Set up attack to make the model approve specific accounts that should be rejected
    # Target class 1 (approval) for applications that should be rejected
    attack = BitFlipAttack(
        model=model,
        dataset=train_dataset,
        target_asr=0.8,
        max_bit_flips=3,  # Very few bits to remain stealthy
        accuracy_threshold=0.1,  # Allow only small drop in overall accuracy
        device=device,
        attack_mode='targeted',
        layer_sensitivity=True,
        hybrid_sensitivity=True,
        alpha=0.5
    )
    
    # Run attack
    print("Performing attack to approve specific high-risk accounts...")
    results = attack.perform_attack(
        target_class=1,  # Target approval class
        num_candidates=100
    )
    
    # Save results
    attack.save_results(results, output_dir="results_pii_attack")
    
    # Demonstrate PII leakage
    print("\nDemonstrating PII leakage impact:")
    
    # Load full dataset with PII
    full_data = pd.read_csv('banking_pii_dataset.csv')
    
    # Get test dataset
    test_dataset = BankingDataset('banking_test.csv', is_train=False)
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
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # If predicted as approved but is high risk
            if predicted.item() == 1:
                idx = test_dataset.data.iloc[i]['customer_id']
                customer = full_data[full_data['customer_id'] == idx].iloc[0]
                
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
    print("\nAttack completed. Results saved to results_pii_attack/")

if __name__ == "__main__":
    run_pii_attack()