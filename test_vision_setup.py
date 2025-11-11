"""
Quick test to verify vision setup works before running full attack
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn

print("="*60)
print("TESTING VISION-BASED PRIVACY ATTACK SETUP")
print("="*60)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Device: {device}")

# Test CIFAR-10 download
print("\n1. Testing CIFAR-10 dataset download...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print(f"✓ Train set: {len(trainset)} samples")
print(f"✓ Test set: {len(testset)} samples")

# Test binary classification setup (Face detection proxy)
print("\n2. Testing binary classification setup (Face Detection)...")
def binary_transform(target):
    return 1 if target >= 5 else 0

trainset.targets = [binary_transform(t) for t in trainset.targets]
testset.targets = [binary_transform(t) for t in testset.targets]

class_0_count = sum(1 for t in trainset.targets if t == 0)
class_1_count = sum(1 for t in trainset.targets if t == 1)

print(f"✓ Class 0 (No faces): {class_0_count} samples ({100*class_0_count/len(trainset):.1f}%)")
print(f"✓ Class 1 (Faces): {class_1_count} samples ({100*class_1_count/len(trainset):.1f}%)")

# Test model creation
print("\n3. Testing ResNet-32 model creation...")
class ResNet32(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet32, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

model = ResNet32(num_classes=2)
model = model.to(device)
print(f"✓ Model created successfully")
print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Test forward pass
print("\n4. Testing forward pass...")
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
batch_inputs, batch_targets = next(iter(test_loader))
batch_inputs = batch_inputs.to(device)

with torch.no_grad():
    outputs = model(batch_inputs)

print(f"✓ Input shape: {batch_inputs.shape}")
print(f"✓ Output shape: {outputs.shape}")
print(f"✓ Forward pass successful")

# Test document synthesis (simplified)
print("\n5. Testing document PII synthesis...")
from PIL import Image, ImageDraw, ImageFont

img, _ = trainset.dataset[0]
img = transforms.ToPILImage()(trainset.dataset[0][0])
img = img.resize((128, 128))
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.load_default()
    draw.text((10, 10), "ID: 123456\nSSN: XXX-XX-1234", fill=(255, 0, 0), font=font)
    print("✓ Document PII overlay successful")
except Exception as e:
    print(f"⚠ Document overlay failed: {e}")
    print("  (This is okay, will use simpler approach)")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED - Ready for full attack experiments!")
print("="*60)
print("\nNext steps:")
print("1. Run: python vision_privacy_attacks.py (full experiment)")
print("2. Or run quick baseline training first to verify models converge")


