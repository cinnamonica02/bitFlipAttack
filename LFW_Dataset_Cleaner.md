# LFW Dataset Cleaner - Google Colab Notebook

**Purpose**: Validate, clean, and fix corrupted LFW images before using for bit-flip attack

Copy each code cell below into Google Colab and run sequentially.

---

## 📦 Cell 1: Setup and Upload

```python
# Install required packages
!pip install Pillow tqdm -q

from google.colab import files
import zipfile
import os
from pathlib import Path

print("Upload your lfw-deepfunneled.zip file (or create zip from your local folder first)")
print("To create zip on Windows: Right-click lfw-deepfunneled folder → Send to → Compressed folder")

# Upload the dataset
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"\nExtracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("✓ Extraction complete!")
```

---

## 🔍 Cell 2: Scan for Corrupted Images

```python
from PIL import Image
from tqdm import tqdm
import os

# Find the extracted directory
lfw_dir = None
for item in os.listdir('.'):
    if 'lfw' in item.lower() and os.path.isdir(item):
        lfw_dir = item
        break

if not lfw_dir:
    print("Error: LFW directory not found. Please check extraction.")
else:
    print(f"Found LFW directory: {lfw_dir}")

# Scan all images
print("\nScanning for corrupted images...")
all_images = []
corrupted_images = []
valid_images = []

for person_name in tqdm(os.listdir(lfw_dir)):
    person_dir = os.path.join(lfw_dir, person_name)
    if os.path.isdir(person_dir):
        for img_file in os.listdir(person_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_dir, img_file)
                all_images.append(img_path)
                
                try:
                    # Try to open and verify
                    with Image.open(img_path) as img:
                        img.verify()  # Check if valid
                    
                    # Re-open for actual validation (verify closes the file)
                    with Image.open(img_path) as img:
                        img.load()  # Force load to check data integrity
                    
                    valid_images.append(img_path)
                    
                except Exception as e:
                    corrupted_images.append((img_path, str(e)))

print(f"\n{'='*60}")
print(f"SCAN RESULTS:")
print(f"{'='*60}")
print(f"Total images found: {len(all_images)}")
print(f"Valid images: {len(valid_images)} ({100*len(valid_images)/len(all_images):.1f}%)")
print(f"Corrupted images: {len(corrupted_images)} ({100*len(corrupted_images)/len(all_images):.1f}%)")
print(f"{'='*60}")

if corrupted_images:
    print(f"\nFirst 20 corrupted images:")
    for img_path, error in corrupted_images[:20]:
        print(f"  {img_path}")
        print(f"    → {error[:100]}")
```

---

## 🛠️ Cell 3: Attempt to Fix/Convert Corrupted Images

```python
from PIL import Image
import os

print("Attempting to repair corrupted images...")
repaired_count = 0
still_corrupted = []

for img_path, error in tqdm(corrupted_images):
    try:
        # Try to open with PIL and re-save
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Re-save with quality 95
            img.save(img_path, 'JPEG', quality=95)
            repaired_count += 1
            
    except Exception as e:
        still_corrupted.append((img_path, str(e)))
        print(f"\n  Cannot repair: {img_path}")
        print(f"    Error: {str(e)[:100]}")

print(f"\n{'='*60}")
print(f"REPAIR RESULTS:")
print(f"{'='*60}")
print(f"Repaired: {repaired_count}/{len(corrupted_images)}")
print(f"Still corrupted: {len(still_corrupted)}")
print(f"{'='*60}")
```

---

## 🗑️ Cell 4: Remove Unrecoverable Corrupted Images

```python
import os

print("Removing unrecoverable corrupted images...")

for img_path, error in still_corrupted:
    try:
        os.remove(img_path)
        print(f"  Removed: {img_path}")
    except Exception as e:
        print(f"  Failed to remove {img_path}: {e}")

print(f"\n✓ Removed {len(still_corrupted)} unrecoverable images")
print(f"✓ Dataset now has {len(valid_images) + repaired_count} clean images")
```

---

## 📊 Cell 5: Verify Cleaned Dataset

```python
from PIL import Image
from tqdm import tqdm

print("Final verification of cleaned dataset...")
final_valid = 0
final_corrupted = 0

for person_name in tqdm(os.listdir(lfw_dir)):
    person_dir = os.path.join(lfw_dir, person_name)
    if os.path.isdir(person_dir):
        for img_file in os.listdir(person_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(img_path) as img:
                        img.load()
                    final_valid += 1
                except:
                    final_corrupted += 1

print(f"\n{'='*60}")
print(f"FINAL DATASET STATUS:")
print(f"{'='*60}")
print(f"Valid images: {final_valid}")
print(f"Still corrupted: {final_corrupted}")
print(f"Success rate: {100*final_valid/(final_valid+final_corrupted):.2f}%")
print(f"{'='*60}")

if final_corrupted == 0:
    print("\n✅ ALL IMAGES VALIDATED - Dataset is clean!")
else:
    print(f"\n⚠ {final_corrupted} images still corrupted. May need manual inspection.")
```

---

## 📥 Cell 6: Download Cleaned Dataset

```python
import shutil
from google.colab import files

# Create zip of cleaned dataset
print("Creating zip of cleaned dataset...")
shutil.make_archive('lfw-deepfunneled-cleaned', 'zip', lfw_dir)

print("Downloading cleaned dataset...")
files.download('lfw-deepfunneled-cleaned.zip')

print(f"\n✅ Download complete!")
print(f"Upload this cleaned dataset to your RunPod instance")
```

---

## 🔧 Alternative Cell 6B: Just Generate File List (Faster)

```python
# Instead of downloading whole dataset, just save list of valid images
import json

valid_image_list = []

for person_name in os.listdir(lfw_dir):
    person_dir = os.path.join(lfw_dir, person_name)
    if os.path.isdir(person_dir):
        for img_file in os.listdir(person_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_dir, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(img_path) as img:
                        img.load()
                    # Store relative path
                    rel_path = os.path.join(person_name, img_file)
                    valid_image_list.append(rel_path)
                except:
                    pass

# Save list
with open('lfw_valid_images.json', 'w') as f:
    json.dump(valid_image_list, f, indent=2)

print(f"✓ Saved {len(valid_image_list)} valid image paths")
files.download('lfw_valid_images.json')

print("\nYou can use this JSON file to filter images on RunPod without re-uploading!")
```

---

## 📝 **Usage Instructions**

1. **On Windows**: Zip your `lfw-deepfunneled` folder
2. **Open Google Colab**: https://colab.research.google.com
3. **Run cells 1-5** to clean the dataset
4. **Choose Cell 6 OR 6B**:
   - **Cell 6**: Downloads entire cleaned dataset (~100MB)
   - **Cell 6B**: Just downloads JSON list (~100KB) - **RECOMMENDED**

**If using Cell 6B** (recommended):
- Upload `lfw_valid_images.json` to your RunPod
- I'll modify the script to use only images from this list
- Much faster than re-uploading entire dataset!

---

**Want me to also create a quick diagnostic script to run on RunPod first to see what's actually wrong with the images?**

