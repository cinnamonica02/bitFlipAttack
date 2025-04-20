import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, ImageFolder

def load_clip_model(dataset_name="cifar10", batch_size=32, model_name="ViT-B/32", pretrained="openai"):
    """
    Load a CLIP model and dataset for bit flipping attack evaluation
    
    Args:
        dataset_name: Name of the dataset to use
        batch_size: Batch size for dataloader
        model_name: CLIP model variant to use
        pretrained: Pretrained model source
        
    Returns:
        model: CLIP model
        dataset: Dataset for evaluation
    """
    try:
        import clip
    except ImportError:
        print("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
        return None, None
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    try:
        model, preprocess = clip.load(model_name, device=device, jit=False)
        print(f"CLIP model {model_name} loaded")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return None, None
    
    # Load and prepare dataset
    if dataset_name.lower() == "cifar10":
        # Define CIFAR-10 transform
        train_dataset = CIFAR10(
            root="./data", 
            train=True,
            download=True, 
            transform=preprocess
        )
        
        test_dataset = CIFAR10(
            root="./data", 
            train=False,
            download=True, 
            transform=preprocess
        )
        
        # Use a small subset for faster evaluation
        eval_dataset = torch.utils.data.Subset(
            test_dataset, 
            indices=range(min(1000, len(test_dataset)))
        )
        
    elif dataset_name.lower() == "imagenet":
        # Assumes ImageNet is available locally
        # Define ImageNet transform
        imagenet_dir = "./data/imagenet"
        
        if not os.path.exists(imagenet_dir):
            print(f"ImageNet data not found at {imagenet_dir}")
            return model, None
            
        # Use validation set for evaluation
        eval_dataset = ImageFolder(
            root=os.path.join(imagenet_dir, "val"),
            transform=preprocess
        )
        
        # Use a small subset for faster evaluation
        eval_dataset = torch.utils.data.Subset(
            eval_dataset, 
            indices=range(min(1000, len(eval_dataset)))
        )
        
    elif dataset_name.lower() == "mscoco":
        # For MSCOCO, we can use image-text pairs
        # This would need a custom dataset implementation
        print("MSCOCO dataset not implemented yet")
        return model, None
        
    else:
        print(f"Unsupported dataset: {dataset_name}")
        return model, None
    
    # Create dataloaders
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4
    )
    
    # Create wrapper for model
    clip_wrapper = CLIPModelWrapper(model, device)
    
    return clip_wrapper, eval_dataset

class CLIPModelWrapper(nn.Module):
    """
    Wrapper for CLIP model to standardize interface for bit flip attack
    """
    def __init__(self, model, device="cuda"):
        super().__init__()
        self.model = model
        self.device = device
        
    def forward(self, inputs):
        """
        Forward pass with images
        
        Args:
            inputs: Image inputs
            
        Returns:
            logits: Image features normalized to unit sphere
        """
        # CLIP returns image features and text features
        # For classification, we just use the image features
        with torch.no_grad():
            image_features = self.model.encode_image(inputs)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def to(self, device):
        """
        Move model to device
        
        Args:
            device: Target device
            
        Returns:
            self: Updated wrapper
        """
        self.device = device
        self.model.to(device)
        return self 