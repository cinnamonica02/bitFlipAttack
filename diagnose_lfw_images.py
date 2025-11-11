"""
Diagnose what's wrong with LFW images
"""
from PIL import Image
import os
from collections import defaultdict

print("="*60)
print("DIAGNOSING LFW IMAGE ISSUES")
print("="*60)

lfw_dir = './data/lfw-deepfunneled'

# Test the specific corrupted images we saw
test_images = [
    './data/lfw-deepfunneled/Igor_Ivanov/Igor_Ivanov_0007.jpg',
    './data/lfw-deepfunneled/Angelina_Jolie/Angelina_Jolie_0014.jpg',
    './data/lfw-deepfunneled/Alisha_Richman/Alisha_Richman_0001.jpg',
    './data/lfw-deepfunneled/Donald_Rumsfeld/Donald_Rumsfeld_0104.jpg',
    './data/lfw-deepfunneled/Bill_Richardson/Bill_Richardson_0001.jpg'
]

error_types = defaultdict(int)

print("\nChecking specific corrupted images:")
print("-"*60)

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"\n❌ {img_path}")
        print(f"   FILE DOES NOT EXIST")
        error_types['missing_file'] += 1
        continue
    
    # Check file size
    file_size = os.path.getsize(img_path)
    print(f"\n🔍 {img_path}")
    print(f"   File size: {file_size} bytes")
    
    # Check if it's actually an image
    try:
        with open(img_path, 'rb') as f:
            header = f.read(20)
            print(f"   File header: {header[:10]}")
            
            # Check for JPEG magic bytes
            if header[:2] == b'\xff\xd8':
                print(f"   ✓ Valid JPEG header")
            else:
                print(f"   ❌ Invalid JPEG header! First bytes: {header[:2]}")
                error_types['invalid_header'] += 1
    except Exception as e:
        print(f"   ❌ Cannot read file: {e}")
        error_types['cannot_read'] += 1
        continue
    
    # Try to open with PIL
    try:
        img = Image.open(img_path)
        print(f"   ✓ PIL can open: mode={img.mode}, size={img.size}")
        
        # Try to load pixels
        try:
            img.load()
            print(f"   ✓ Image data loads successfully")
            error_types['actually_valid'] += 1
        except Exception as e:
            print(f"   ❌ Image data corrupted: {e}")
            error_types['corrupted_data'] += 1
            
    except Exception as e:
        print(f"   ❌ PIL cannot open: {e}")
        error_types['pil_cannot_open'] += 1

print("\n" + "="*60)
print("ERROR TYPE SUMMARY:")
print("="*60)
for error_type, count in sorted(error_types.items()):
    print(f"{error_type}: {count}")
print("="*60)

# Now scan ALL images
print("\n\nScanning ALL LFW images...")
total_scanned = 0
total_corrupted = 0
corruption_details = defaultdict(int)

for person_name in os.listdir(lfw_dir):
    person_dir = os.path.join(lfw_dir, person_name)
    if os.path.isdir(person_dir):
        for img_file in os.listdir(person_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_dir, img_file)
                total_scanned += 1
                
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    # Re-open after verify
                    with Image.open(img_path) as img:
                        img.load()
                except Exception as e:
                    total_corrupted += 1
                    error_msg = str(e)
                    if 'cannot identify' in error_msg:
                        corruption_details['cannot_identify'] += 1
                    elif 'truncated' in error_msg or 'incomplete' in error_msg:
                        corruption_details['truncated'] += 1
                    elif 'decode' in error_msg:
                        corruption_details['decode_error'] += 1
                    else:
                        corruption_details['other'] += 1

print(f"\n{'='*60}")
print(f"FULL DATASET SCAN RESULTS:")
print(f"{'='*60}")
print(f"Total images scanned: {total_scanned}")
print(f"Corrupted images: {total_corrupted} ({100*total_corrupted/total_scanned:.2f}%)")
print(f"Valid images: {total_scanned - total_corrupted} ({100*(total_scanned-total_corrupted)/total_scanned:.2f}%)")
print(f"\nCorruption breakdown:")
for error_type, count in sorted(corruption_details.items()):
    print(f"  {error_type}: {count}")
print(f"{'='*60}")

# Recommendation
if total_corrupted / total_scanned > 0.10:
    print("\n⚠ MORE THAN 10% CORRUPTED - Re-download dataset recommended")
elif total_corrupted / total_scanned > 0.01:
    print("\n⚠ 1-10% CORRUPTED - Can proceed but filter out bad images")
else:
    print("\n✅ <1% CORRUPTED - Dataset is good, just filter during loading")

