import os
import sys
import torch
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VQADataset(Dataset):
    """
    Dataset for Visual Question Answering
    """
    def __init__(self, processor, questions, images, answers=None, max_length=512):
        self.processor = processor
        self.questions = questions
        self.images = images
        self.answers = answers
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        image_path = self.images[idx]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Process inputs
        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Remove batch dimension
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                inputs[k] = v.squeeze(0)
        
        # Add labels if available
        if self.answers is not None:
            inputs["labels"] = torch.tensor(self.answers[idx])
            
        return inputs

def load_vqav2_dataset(processor, split="val", max_samples=1000):
    """
    Load VQAv2 dataset
    
    Args:
        processor: Model processor/tokenizer
        split: Dataset split (train/val/test)
        max_samples: Maximum number of samples to load
        
    Returns:
        dataset: VQADataset for evaluation
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the 'datasets' library: pip install datasets")
        return None
    
    # Load VQAv2 dataset
    try:
        # Use Hugging Face datasets
        dataset = load_dataset("vqa_v2", split=split)
        
        # Convert to our format
        questions = []
        images = []
        answers = []
        
        # Limit number of samples
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
                
            questions.append(item["question"])
            
            # Download image if needed
            img_id = item["image_id"]
            img_path = f"path_to_vqa_images/{split}2014/COCO_{split}2014_{img_id:012d}.jpg"
            images.append(img_path)
            
            # Use the most common answer as the label
            if "answers" in item:
                answer_counts = {}
                for ans in item["answers"]:
                    ans_text = ans["answer"]
                    answer_counts[ans_text] = answer_counts.get(ans_text, 0) + 1
                
                # Get the most common answer
                most_common = max(answer_counts.items(), key=lambda x: x[1])[0]
                answers.append(most_common)
        
        return VQADataset(processor, questions, images, answers)
    
    except Exception as e:
        print(f"Error loading VQAv2 dataset: {e}")
        return None

def load_textvqa_dataset(processor, split="val", max_samples=1000):
    """
    Load TextVQA dataset
    
    Args:
        processor: Model processor/tokenizer
        split: Dataset split (train/val/test)
        max_samples: Maximum number of samples to load
        
    Returns:
        dataset: VQADataset for evaluation
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the 'datasets' library: pip install datasets")
        return None
    
    # Load TextVQA dataset
    try:
        # Use Hugging Face datasets
        dataset = load_dataset("textvqa", split=split)
        
        # Convert to our format
        questions = []
        images = []
        answers = []
        
        # Limit number of samples
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
                
            questions.append(item["question"])
            
            # Download image if needed
            img_id = item["image_id"]
            img_path = f"path_to_textvqa_images/{img_id}.jpg"
            images.append(img_path)
            
            # Use the most common answer as the label
            if "answers" in item:
                answer_counts = {}
                for ans in item["answers"]:
                    answer_counts[ans] = answer_counts.get(ans, 0) + 1
                
                # Get the most common answer
                most_common = max(answer_counts.items(), key=lambda x: x[1])[0]
                answers.append(most_common)
        
        return VQADataset(processor, questions, images, answers)
    
    except Exception as e:
        print(f"Error loading TextVQA dataset: {e}")
        return None

def load_vlm_model(model_name="llava-hf/llava-1.5-7b-hf", dataset_name="vqav2", device="cuda"):
    """
    Load a vision-language model for bit flip attack evaluation
    
    Args:
        model_name: Hugging Face model name or path
        dataset_name: Name of the dataset to use (vqav2, textvqa)
        device: Device to load model on
        
    Returns:
        model: Loaded model
        dataset: Evaluation dataset
    """
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        print("Please install transformers: pip install transformers")
        return None, None
    
    print(f"Loading VLM model: {model_name}")
    
    try:
        # Load processor and model
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Error loading VLM model: {e}")
        return None, None
    
    # Load dataset based on name
    dataset = None
    if dataset_name.lower() == "vqav2":
        dataset = load_vqav2_dataset(processor)
    elif dataset_name.lower() == "textvqa":
        dataset = load_textvqa_dataset(processor)
    else:
        print(f"Unsupported dataset: {dataset_name}")
        return model, None
    
    # Create model wrapper
    model_wrapper = VLMAttackWrapper(model, processor, device)
    
    return model_wrapper, dataset

class VLMAttackWrapper:
    """
    Wrapper for VLM models to standardize interface for bit flip attack
    """
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        
    def __call__(self, x):
        """
        Forward pass with input batch
        
        Args:
            x: Dictionary with input_ids, attention_mask, and pixel_values
            
        Returns:
            logits: Prediction logits
        """
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in x.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Return the logits
        return outputs.logits
    
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