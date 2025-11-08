"""
Real-world example: PII exposure through bit flip attack on a PII detection model
"""
import sys
import pickle
from datetime import datetime
import io
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack
from sklearn.feature_extraction.text import CountVectorizer

class Logger:
    def __init__(self, log_dir="results_pii_attack"):
        self.terminal = sys.stdout
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"{log_dir}/console_output_{timestamp}.log", "w")
        self.output_buffer = io.StringIO()
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.output_buffer.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def save_results(self, results_dir="results_pii_attack"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'console_output': self.output_buffer.getvalue(),
            'timestamp': timestamp
        }
        with open(f"{results_dir}/results_{timestamp}.pkl", 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {results_dir}/results_{timestamp}.pkl")
        
    def __del__(self):
        self.log_file.close()
        self.output_buffer.close()

class PIIDataset(Dataset):
    vectorizer = None  # Class-level vectorizer to share between instances
    
    def __init__(self, csv_file, is_train=True):
        self.data = pd.read_csv(csv_file)
        
        # Convert text to numerical features using simple bag of words
        if PIIDataset.vectorizer is None and is_train:
            # Only fit vectorizer on training data
            PIIDataset.vectorizer = CountVectorizer(max_features=100)
            self.X = PIIDataset.vectorizer.fit_transform(self.data['text']).toarray()
        elif PIIDataset.vectorizer is not None:
            # Use already fitted vectorizer for test data
            self.X = PIIDataset.vectorizer.transform(self.data['text']).toarray()
        else:
            raise ValueError("Must create training dataset before test dataset")
            
        self.y = self.data['contains_pii'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        # Return as dict to match evaluation function expectations
        # Use dummy attention_mask since this simple model doesn't use it
        return {
            'input_ids': features,
            'attention_mask': torch.ones(features.shape[0], dtype=torch.long),
            'labels': label
        }

class PIIDetectionModel(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 classes: contains PII or not
        
    def forward(self, x=None, input_ids=None, attention_mask=None, **kwargs):
        # Accept both simple tensor input and transformer-style kwargs
        if input_ids is not None:
            x = input_ids
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Train or load model
def train_model(train_csv='data/pii_dataset_train.csv', epochs=10):
    train_dataset = PIIDataset(train_csv)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = PIIDetectionModel()
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
        
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
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
    torch.save(model.state_dict(), 'models_dir/pii_model.pth')
    
    torch.save(model.state_dict(), 'temp_original_model.pt')
    
    print("Model trained and saved.")
    return model, train_dataset

# Add this function after train_model and before run_pii_attack
def evaluate_model(model, test_dataset, device):
    """Evaluate model performance on test dataset"""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    pii_detected = 0
    pii_total = 0
    
    print("\nEvaluating model performance:")
    print("-" * 80)
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Track PII detection specifically
            pii_mask = labels == 1
            pii_total += pii_mask.sum().item()
            pii_detected += (predicted[pii_mask] == 1).sum().item()
    
    accuracy = 100. * correct / total
    pii_accuracy = 100. * pii_detected / pii_total if pii_total > 0 else 0
    
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"PII Detection Rate: {pii_accuracy:.2f}%")
    print(f"Total samples: {total}, PII samples: {pii_total}")
    print("-" * 80)
    
    return accuracy, pii_accuracy

# 4. Bit flip attack
def run_pii_attack():
    print("=" * 80)
    print("Real-world Example: PII Exposure through Bit Flip Attack on PII Detection Model")
    print("=" * 80)
    
    # Check if model exists, otherwise train it
    if os.path.exists('models_dir/pii_model.pth'):
        print("Loading pre-trained model...")
        model = PIIDetectionModel()
        model.load_state_dict(torch.load('models_dir/pii_model.pth'))
        train_dataset = PIIDataset('data/pii_dataset_train.csv')
    else:
        model, train_dataset = train_model(epochs=5)  # Quick training
    
    # Create results directory
    os.makedirs("results_pii_attack", exist_ok=True)
    
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Get test dataset
    test_dataset = PIIDataset('data/pii_dataset_test.csv', is_train=False)
    
    # Evaluate model before attack
    print("\nBaseline Performance (Before Attack):")
    original_accuracy, original_pii_rate = evaluate_model(model, test_dataset, device)
    
    # Save original model state
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Set up attack to make the model misclassify PII as non-PII
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
    
    # Run attack targeting class 0 (no PII) for samples that contain PII
    print("\nPerforming attack to misclassify PII as non-PII...")
    results = attack.perform_attack(
        target_class=0,  # Target no-PII class
        num_candidates=100
    )
    
    # Save attack results
    attack.save_results(results, output_dir="results_pii_attack")
    
    # Evaluate model after attack
    print("\nModel Performance After Attack:")
    attacked_accuracy, attacked_pii_rate = evaluate_model(model, test_dataset, device)
    
    # Show impact
    print("\nAttack Impact Summary:")
    print("-" * 80)
    print(f"Overall Accuracy: {original_accuracy:.2f}% -> {attacked_accuracy:.2f}% (Change: {attacked_accuracy - original_accuracy:.2f}%)")
    print(f"PII Detection Rate: {original_pii_rate:.2f}% -> {attacked_pii_rate:.2f}% (Change: {attacked_pii_rate - original_pii_rate:.2f}%)")
    print("-" * 80)
    
    # Demonstrate specific examples of PII leakage
    print("\nExample texts now failing PII detection:")
    print("-" * 80)
    print(f"{'Text':<60} {'True Label':<12} {'Prediction':<10}")
    print("-" * 80)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels']
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # If predicted as non-PII but contains PII
            if predicted.item() == 0 and labels.item() == 1:
                text = test_dataset.data.iloc[i]['text']
                count += 1
                print(f"{text[:57] + '...':<60} {'Contains PII':<12} {'No PII':<10}")
                
                if count >= 5:  # Show just a few examples
                    break
    
    print("-" * 80)
    print(f"The attack has caused the model to misclassify texts containing PII as non-PII.")
    print(f"In a real-world scenario, this could lead to accidental exposure of sensitive information")
    print(f"such as SSNs, medical records, or financial data.")
    print("\nAttack completed. Results saved to results_pii_attack/")

if __name__ == "__main__":
    logger = Logger()
    sys.stdout = logger
    try:
        run_pii_attack()
        # Save console output and results
        logger.save_results()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
    finally:
        # Restore original stdout
        sys.stdout = logger.terminal