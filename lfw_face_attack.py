"""
Bit-Flip Attack on Face Detection Model (LFW Dataset)
======================================================

Test script to validate vision-based bit-flip attacks work before extending to documents.

Task: Binary Face Detection
- Class 0: No face (background images from CIFAR-10)
- Class 1: Face present (images from LFW dataset)

Attack Goal: Cause face detector to miss faces (privacy violation)

Based on:
- Groan: "Tossing in the Dark" (USENIX Security 2024)
- Aegis: Defense against targeted bit-flip attacks
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from PIL import Image
from sklearn.datasets import fetch_lfw_people

# Import attack classes
from bitflip_attack.attacks.umup_bit_flip_attack import UmupBitFlipAttack
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack


class ResNet32(nn.Module):
    """ResNet-32 for binary face detection"""
    def __init__(self, num_classes=2):
        super(ResNet32, self).__init__()
        # Use ResNet-18 as base (closest to ResNet-32 in torchvision)
        self.resnet = models.resnet18(pretrained=False)
        # Modify first conv for smaller images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for smaller images
        # Change final layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class LFWFaceDataset(Dataset):
    """
    LFW Face Dataset wrapper
    Returns faces from LFW as Class 1 (face present)
    """
    def __init__(self, transform=None, data_dir='./data/lfw'):
        self.transform = transform
        self.data_dir = data_dir
        
        print(f"Loading LFW dataset from {data_dir}...")
        
        # Try to load from directory structure first
        if os.path.exists(data_dir):
            self.images = []
            self.labels = []
            corrupted_count = 0
            
            # Walk through LFW directory and validate images
            print("Validating LFW images...")
            for person_name in os.listdir(data_dir):
                person_dir = os.path.join(data_dir, person_name)
                if os.path.isdir(person_dir):
                    for img_file in os.listdir(person_dir):
                        if img_file.endswith(('.jpg', '.png', '.jpeg')):
                            img_path = os.path.join(person_dir, img_file)
                            # Validate image can be opened
                            try:
                                test_img = Image.open(img_path)
                                test_img.verify()  # Check if it's a valid image
                                self.images.append(img_path)
                                self.labels.append(1)  # Face present
                            except Exception as e:
                                corrupted_count += 1
                                if corrupted_count <= 10:  # Print first 10
                                    print(f"  Skipping corrupted: {img_path}")
                                    print(f"    Error: {e}")
            
            print(f"✓ Loaded {len(self.images)} valid face images from LFW directory")
            if corrupted_count > 0:
                print(f"⚠ Skipped {corrupted_count} corrupted/invalid images during loading")
        
        else:
            print(f"LFW directory not found at {data_dir}")
            print("Attempting to download using sklearn...")
            try:
                lfw_data = fetch_lfw_people(data_home='./data', min_faces_per_person=1, 
                                           resize=0.5, color=True)
                self.images = lfw_data.images
                self.labels = [1] * len(self.images)  # All are faces
                self.is_sklearn = True
                print(f"✓ Downloaded {len(self.images)} face images using sklearn")
            except Exception as e:
                print(f"Error downloading LFW: {e}")
                raise
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image with error handling for corrupted files
        try:
            if isinstance(self.images[idx], str):
                # Load from file path
                image = Image.open(self.images[idx]).convert('RGB')
            else:
                # Already numpy array from sklearn
                image = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
            
            if self.transform:
                image = self.transform(image)
            
            label = self.labels[idx]
            return image, label
        except Exception as e:
            # Skip corrupted images by returning a black placeholder
            print(f"Warning: Skipping corrupted image {self.images[idx]}: {e}")
            # Return a black image as placeholder
            black_image = torch.zeros(3, 64, 64)  # Assuming 64x64 size
            return black_image, self.labels[idx]


class NonFaceDataset(Dataset):
    """
    Non-face dataset using CIFAR-10 images (no people)
    Returns as Class 0 (no face)
    """
    def __init__(self, transform=None, data_dir='./data'):
        self.transform = transform
        
        print("Loading CIFAR-10 for non-face images...")
        # Load CIFAR-10 but exclude classes with people
        # CIFAR-10 classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 
        #                   5=dog, 6=frog, 7=horse, 8=ship, 9=truck
        # We'll use classes 0,1,8,9 (vehicles, no living things that look face-like)
        
        cifar_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                                                   download=True, transform=None)
        
        # Filter to non-animal classes
        non_face_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
        self.images = []
        self.labels = []
        
        for img, label in cifar_data:
            if label in non_face_classes:
                self.images.append(img)
                self.labels.append(0)  # No face
        
        print(f"✓ Loaded {len(self.images)} non-face images from CIFAR-10")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def create_face_detection_dataloaders(batch_size=32, data_dir='./data', img_size=64):
    """
    Create balanced dataloaders for face detection
    
    Returns:
        train_loader, test_loader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("\n" + "="*60)
    print("Creating Face Detection Dataset")
    print("="*60)
    
    # Load face dataset (LFW)
    try:
        face_dataset = LFWFaceDataset(transform=transform, data_dir=os.path.join(data_dir, 'lfw-deepfunneled'))
    except Exception as e:
        print(f"Failed to load LFW: {e}")
        print("Falling back to alternative...")
        # Fallback: Use CIFAR-10 classes with people
        print("Using CIFAR-10 as fallback (not ideal but works for testing)")
        cifar_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                                                   download=True, transform=transform)
        # Use classes 2,3,4,5,6,7 as "face-like" (animals)
        face_images = [(img, 1) for img, label in cifar_data if label in [2,3,4,5,6,7]]
        
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        face_dataset = SimpleDataset(face_images)
    
    # Load non-face dataset (CIFAR-10 vehicles)
    non_face_dataset = NonFaceDataset(transform=transform, data_dir=data_dir)
    
    # Balance datasets (take minimum length)
    min_len = min(len(face_dataset), len(non_face_dataset))
    print(f"\nBalancing datasets to {min_len} samples per class")
    
    # Create balanced subsets
    face_indices = torch.randperm(len(face_dataset))[:min_len].tolist()
    non_face_indices = torch.randperm(len(non_face_dataset))[:min_len].tolist()
    
    face_subset = torch.utils.data.Subset(face_dataset, face_indices)
    non_face_subset = torch.utils.data.Subset(non_face_dataset, non_face_indices)
    
    # Combine datasets
    combined_dataset = ConcatDataset([face_subset, non_face_subset])
    
    print(f"Total dataset size: {len(combined_dataset)} ({min_len} faces + {min_len} non-faces)")
    
    # Split into train/test (80/20)
    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train set: {train_size} samples")
    print(f"Test set: {test_size} samples")
    print("="*60 + "\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_face_detector(model, train_loader, test_loader, epochs=15, 
                       device='cuda', target_accuracy=0.80):
    """
    Train face detection model
    Target accuracy 75-85% (realistic for bit-flip attack)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print("\n" + "="*60)
    print("Training Face Detection Model")
    print("="*60)
    print(f"Target accuracy range: 75-85% (realistic for attack)")
    print(f"Device: {device}")
    print()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_face_correct = 0
        val_face_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Track face detection specifically
                face_mask = (targets == 1)
                val_face_total += face_mask.sum().item()
                val_face_correct += (predicted[face_mask] == 1).sum().item()
        
        val_acc = val_correct / val_total
        face_recall = val_face_correct / val_face_total if val_face_total > 0 else 0
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {100*train_acc:.2f}%')
        print(f'  Val Acc: {100*val_acc:.2f}% | Face Recall: {100*face_recall:.2f}%')
        
        # Early stopping if we reach target accuracy range
        if target_accuracy <= val_acc < 0.88:
            print(f"\n✓ Reached target accuracy range ({100*val_acc:.2f}%)")
            print("  Stopping to preserve decision boundaries for attack")
            best_acc = val_acc
            break
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        scheduler.step()
        print()
    
    print(f"Training complete. Best accuracy: {100*best_acc:.2f}%")
    print("="*60 + "\n")
    
    return model, best_acc


def evaluate_face_detector(model, test_loader, device='cuda'):
    """Evaluate face detector with detailed metrics"""
    model.eval()
    model.to(device)
    
    total = 0
    correct = 0
    
    # Face-specific metrics
    face_total = 0
    face_detected = 0  # True positives
    face_missed = 0    # False negatives (PRIVACY RISK!)
    
    # Non-face metrics
    non_face_total = 0
    non_face_correct = 0
    false_alarms = 0   # False positives
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Face metrics (Class 1)
            face_mask = (targets == 1)
            face_total += face_mask.sum().item()
            face_detected += ((predicted == 1) & face_mask).sum().item()
            face_missed += ((predicted == 0) & face_mask).sum().item()
            
            # Non-face metrics (Class 0)
            non_face_mask = (targets == 0)
            non_face_total += non_face_mask.sum().item()
            non_face_correct += ((predicted == 0) & non_face_mask).sum().item()
            false_alarms += ((predicted == 1) & non_face_mask).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    face_recall = face_detected / face_total if face_total > 0 else 0
    privacy_leak_rate = face_missed / face_total if face_total > 0 else 0
    
    print("\n" + "="*60)
    print("Face Detector Evaluation")
    print("="*60)
    print(f"Overall Accuracy: {100*accuracy:.2f}%")
    print(f"Face Detection Rate (Recall): {100*face_recall:.2f}%")
    print(f"🚨 Privacy Leak Rate (Missed Faces): {100*privacy_leak_rate:.2f}%")
    print(f"   ({face_missed}/{face_total} faces missed)")
    print(f"False Alarm Rate: {100*false_alarms/non_face_total if non_face_total > 0 else 0:.2f}%")
    print("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'face_recall': face_recall,
        'privacy_leak_rate': privacy_leak_rate,
        'faces_missed': face_missed,
        'total_faces': face_total
    }


def quantize_model(model, calibration_loader, device='cuda'):
    """Quantize model to 8-bit (as per Groan/Aegis papers)"""
    print("\n" + "="*60)
    print("Quantizing Model to 8-bit")
    print("="*60)
    
    model.eval()
    model.to('cpu')  # Quantization works on CPU
    
    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with sample data
    print("Calibrating quantization...")
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model(inputs)
            break  # One batch is enough
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    print("✓ Model quantized to 8-bit")
    print("="*60 + "\n")
    
    return model


def main():
    """
    Main execution: Train face detector and run bit-flip attacks
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/lfw_face_attack_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("BIT-FLIP ATTACK ON FACE DETECTION MODEL")
    print("="*80)
    print("\nScenario: Social media platform's face detection for privacy protection")
    print("Attack Goal: Cause face detector to miss faces → privacy violation")
    print("="*80 + "\n")
    
    # Step 1: Load dataset
    train_loader, test_loader = create_face_detection_dataloaders(
        batch_size=64, 
        data_dir='./data',
        img_size=64
    )
    
    # Step 2: Train model
    model = ResNet32(num_classes=2)
    model, train_acc = train_face_detector(
        model, train_loader, test_loader,
        epochs=15, device=device, target_accuracy=0.78
    )
    
    # Step 3: Evaluate baseline
    print("📊 BASELINE MODEL EVALUATION")
    baseline_metrics = evaluate_face_detector(model, test_loader, device)
    
    # Save baseline model
    torch.save(model.state_dict(), results_dir / 'face_detector_baseline.pth')
    
    # Step 4: Quantize model
    model_quantized = quantize_model(model, test_loader, device)
    
    # Step 5: Evaluate quantized model
    print("📊 QUANTIZED MODEL EVALUATION")
    quantized_metrics = evaluate_face_detector(model_quantized, test_loader, device)
    
    print("\n" + "="*80)
    print("🎯 READY FOR BIT-FLIP ATTACK")
    print("="*80)
    print("\nNext step: Run bit-flip attack to increase privacy leak rate")
    print(f"Current privacy leak rate: {100*quantized_metrics['privacy_leak_rate']:.2f}%")
    print(f"Target after attack: ≥85% privacy leak rate")
    print(f"\nTo run attack, uncomment the attack code below and execute")
    print("="*80)
    
    # Save metrics
    results = {
        'baseline': baseline_metrics,
        'quantized': quantized_metrics,
        'model': 'ResNet-32',
        'dataset': 'LFW + CIFAR-10',
        'quantization': '8-bit',
        'timestamp': timestamp
    }
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {results_dir}")
    print(f"✓ Baseline model saved")
    
    # TODO: Uncomment to run actual attack
    # print("\n" + "="*80)
    # print("RUNNING BIT-FLIP ATTACK")
    # print("="*80)
    # 
    # attack = UmupBitFlipAttack(
    #     model=model_quantized,
    #     dataset=test_loader.dataset,
    #     target_asr=0.85,
    #     max_bit_flips=20,
    #     accuracy_threshold=0.05,
    #     device=device
    # )
    # 
    # results = attack.perform_attack(target_class=0)  # Make faces → non-faces
    # attack.save_results(results, results_dir)
    # 
    # # Evaluate after attack
    # attacked_metrics = evaluate_face_detector(model_quantized, test_loader, device)
    # print(f"\n🚨 Privacy leak rate increased from {100*baseline_metrics['privacy_leak_rate']:.2f}% "
    #       f"to {100*attacked_metrics['privacy_leak_rate']:.2f}%!")


if __name__ == "__main__":
    main()

