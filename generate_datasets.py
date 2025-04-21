from bitflip_attack.datasets import generate_all_datasets
import os

def main():
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("Starting dataset generation...")

    dataset_paths = generate_all_datasets(
        base_path=data_dir,
        num_records=1000
    )
    
    print("\nGenerated datasets:")
    for dataset_type, path in dataset_paths.items():
        print(f"- {dataset_type}: {path}")

if __name__ == "__main__":
    main() 