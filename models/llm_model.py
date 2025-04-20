import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaForCausalLM, LlamaTokenizer,
    AutoModelForCausalLM, AutoTokenizer,
    PreTrainedModel
)

class LLMDataset(Dataset):
    """
    Dataset for evaluating LLMs on various benchmarks
    """
    def __init__(self, tokenizer, texts, labels, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label)
        }

def create_mmlu_dataset(tokenizer, task_name="astronomy", split="validation", n_shot=5):
    """
    Create a dataset for MMLU benchmark
    
    Args:
        tokenizer: Tokenizer for the model
        task_name: Name of the MMLU task
        split: Dataset split (train/validation/test)
        n_shot: Number of examples to include in few-shot setting
        
    Returns:
        dataset: LLMDataset for evaluation
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the 'datasets' library: pip install datasets")
        return None
    
    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", task_name)
    
    # Get the requested split
    if split not in dataset:
        print(f"Split {split} not found in dataset. Available splits: {dataset.keys()}")
        return None
    
    data_split = dataset[split]
    
    # Format examples (few-shot)
    if n_shot > 0 and "train" in dataset:
        # Get few-shot examples from training set
        train_samples = dataset["train"].select(range(min(n_shot, len(dataset["train"]))))
        
        # Format each evaluation example with few-shot examples
        texts = []
        for example in data_split:
            prompt = "Answer the following multiple-choice questions.\n\n"
            
            # Add few-shot examples
            for i, train_ex in enumerate(train_samples):
                q = train_ex["question"]
                a = train_ex["choices"][train_ex["answer"]]
                prompt += f"Q: {q}\nA: {a}\n\n"
            
            # Add the test question
            prompt += f"Q: {example['question']}\nA:"
            texts.append(prompt)
    else:
        # Zero-shot prompting
        texts = [f"Answer the following multiple-choice question:\n\nQ: {ex['question']}\nA:" for ex in data_split]
    
    # Extract labels
    labels = [ex["answer"] for ex in data_split]
    
    return LLMDataset(tokenizer, texts, labels)

def load_llm_model(model_name="meta-llama/Meta-Llama-3-8B", task="mmlu:astronomy", device="cuda"):
    """
    Load a large language model for bit flip attack evaluation
    
    Args:
        model_name: Hugging Face model name or path
        task: Evaluation task (format: benchmark:subtask)
        device: Device to load model on
        
    Returns:
        model: Loaded model
        dataset: Evaluation dataset
    """
    try:
        # Check if transformers is installed
        import transformers
    except ImportError:
        print("Please install transformers: pip install transformers")
        return None, None
    
    print(f"Loading model: {model_name}")
    
    # Determine model type and load appropriate tokenizer and model
    tokenizer = None
    model = None
    
    if "llama" in model_name.lower():
        try:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(
                model_name, 
                device_map=device,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Error loading LLaMA model: {e}")
            # Fallback to AutoTokenizer/AutoModel
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                print(f"Error loading model with AutoModel: {e}")
                return None, None
    else:
        # Use AutoTokenizer/AutoModel for other models
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Error loading model with AutoModel: {e}")
            return None, None
    
    # Load dataset based on task
    dataset = None
    
    # Parse task string (format: "benchmark:subtask")
    parts = task.split(":")
    benchmark = parts[0].lower()
    subtask = parts[1] if len(parts) > 1 else None
    
    if benchmark == "mmlu":
        dataset = create_mmlu_dataset(tokenizer, task_name=subtask or "astronomy")
    else:
        print(f"Unsupported benchmark: {benchmark}")
        return model, None
    
    return model, dataset

class LLMAttackWrapper:
    """
    Wrapper for LLM models to standardize interface for bit flip attack
    """
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def __call__(self, x):
        """
        Forward pass with input batch
        
        Args:
            x: Dictionary with input_ids and attention_mask
            
        Returns:
            logits: Prediction logits
        """
        input_ids = x["input_ids"].to(self.device)
        attention_mask = x["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
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