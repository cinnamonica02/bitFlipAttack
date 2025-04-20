import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class QuantizedModelWrapper:
    """
    Wrapper for quantized models to standardize interface for bit flip attack
    """
    def __init__(self, model, config, device="cuda"):
        self.model = model
        self.config = config
        self.device = device
        self.quantization_bits = config.get("bits", 8)
        
    def __call__(self, x):
        """
        Forward pass with input batch
        
        Args:
            x: Input data
            
        Returns:
            outputs: Model outputs
        """
        # Move inputs to device if needed
        if isinstance(x, dict):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in x.items()}
        else:
            inputs = x.to(self.device) if isinstance(x, torch.Tensor) else x
        
        with torch.no_grad():
            outputs = self.model(inputs)
            
        return outputs
    
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

def load_bitsandbytes_quantized_model(model_name, bits=8, device="cuda"):
    """
    Load a model quantized with bitsandbytes
    
    Args:
        model_name: Hugging Face model name or path
        bits: Quantization bits (4 or 8)
        device: Device to load model on
        
    Returns:
        model: Loaded quantized model
    """
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("Please install bitsandbytes: pip install bitsandbytes")
        return None
    
    print(f"Loading {bits}-bit quantized model: {model_name}")
    
    # Configure quantization
    if bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # normalized float 4
        )
    else:
        print(f"Unsupported quantization bits: {bits}")
        return None
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            quantization_config=quantization_config
        )
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        return None, None
    
    # Create wrapper with quantization config
    model_wrapper = QuantizedModelWrapper(
        model, 
        config={"bits": bits, "quantization_method": "bitsandbytes"},
        device=device
    )
    
    return model_wrapper, tokenizer

def load_bitnet_model(model_name="microsoft/BitNet-b1.58-1.5B", device="cuda"):
    """
    Load a BitNet model for ultra-low precision evaluation
    
    Args:
        model_name: Hugging Face model name or path
        device: Device to load model on
        
    Returns:
        model: Loaded BitNet model
        tokenizer: Corresponding tokenizer
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Please install transformers: pip install transformers")
        return None, None
    
    print(f"Loading BitNet model: {model_name}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error loading BitNet model: {e}")
        return None, None
    
    # Create wrapper with BitNet config
    model_wrapper = QuantizedModelWrapper(
        model, 
        config={"bits": 1.58, "quantization_method": "bitnet"},
        device=device
    )
    
    return model_wrapper, tokenizer

def load_quantized_model(model_name, quantization="w8", model_type="llm", device="cuda"):
    """
    Load a quantized model for bit flip attack evaluation
    
    Args:
        model_name: Model name or path
        quantization: Quantization type (w8, w4, w1.58)
        model_type: Model type (llm, vlm)
        device: Device to load model on
        
    Returns:
        model: Loaded quantized model
        tokenizer: Corresponding tokenizer
    """
    print(f"Loading {quantization} quantized model: {model_name}")
    
    # Handle BitNet models
    if quantization.lower() == "w1.58" or "bitnet" in model_name.lower():
        return load_bitnet_model(model_name, device)
    
    # Handle Int8/Int4 models
    bits = 8
    if quantization.lower() == "w4":
        bits = 4
    elif quantization.lower() == "w8":
        bits = 8
    
    return load_bitsandbytes_quantized_model(model_name, bits, device) 