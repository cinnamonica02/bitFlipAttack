"""
Bit manipulation module for bit flip attacks

This module contains functions to select bit candidates and perform bit flips
on model parameters.
"""
import torch
import numpy as np
from bitflip_attack.attacks.sensitivity import compute_sensitivity


def select_bit_candidates(model, dataset, layer, n_candidates=1000, 
                         device='cuda', hybrid_sensitivity=True, alpha=0.5,
                         custom_forward_fn=None):
    """
    Select candidate bits for flipping based on sensitivity.
    
    Args:
        model: The model being attacked
        dataset: Dataset for evaluating model performance
        layer: Layer information dictionary
        n_candidates: Number of bit candidates to select
        device: Device to run computations on
        hybrid_sensitivity: Whether to use hybrid sensitivity metric
        alpha: Weight for hybrid sensitivity metric
        custom_forward_fn: Custom forward function for the model
        
    Returns:
        candidates: List of candidate bits (indices, positions)
    """
    # Get a batch of data for gradient computation
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    
    try:
        inputs, targets = next(iter(dataloader))
        if isinstance(inputs, dict):
            # Handle dictionary inputs (common in huggingface datasets)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)
        
        if isinstance(targets, dict):
            targets = {k: v.to(device) for k, v in targets.items()}
        else:
            targets = targets.to(device)
    except:
        # If dataset structure is different, try unpacking with custom logic
        batch = next(iter(dataloader))
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)
        else:
            raise ValueError("Unable to extract inputs and targets from dataset")
    
    # Compute sensitivity
    sensitivity = compute_sensitivity(
        model, layer, inputs, targets, 
        use_gradient=hybrid_sensitivity, 
        use_magnitude=True,
        alpha=alpha,
        custom_forward_fn=custom_forward_fn
    )
    
    # Get top-k indices by sensitivity
    n_candidates = min(n_candidates, layer['n_params'])
    _, indices = torch.topk(sensitivity.view(-1), n_candidates)
    
    # For each parameter, consider different bit positions (focus on MSBs)
    candidates = []
    
    # Determine bit width based on parameter type
    bit_width = determine_bit_width(layer)
    
    # For each selected parameter, add MSBs as candidates
    for idx in indices:
        # Convert flat index to tensor coordinates
        flat_idx = idx.item()
        coords = np.unravel_index(flat_idx, layer['shape'])
        
        # Add MSB and sign bit as candidates
        bit_positions = get_relevant_bit_positions(bit_width)
        
        for bit_pos in bit_positions:
            candidates.append({
                'layer_name': layer['name'],
                'layer_idx': layer['module_idx'] if 'module_idx' in layer else -1,
                'parameter_idx': flat_idx,
                'coords': coords,
                'bit_position': bit_pos,
                'sensitivity': sensitivity.view(-1)[flat_idx].item()
            })
    
    return candidates


def determine_bit_width(layer):
    """
    Determine the bit width of parameters in a layer.
    
    Args:
        layer: Layer information dictionary
        
    Returns:
        bit_width: Number of bits per parameter
    """
    # Default to float32 (typical case)
    bit_width = 32
    
    # Determine actual bit width of parameters
    if hasattr(layer['module'], 'weight_fake_quant'):
        # For quantized models with fake quantization
        bit_width = layer['module'].weight_fake_quant.bit_width
    elif layer.get('is_quantized', False):
        # For models with explicit quantization
        bit_width = 8  # Assume int8 for quantized layers
    elif layer['module'].weight.dtype == torch.float16:
        bit_width = 16  # FP16
    elif layer['module'].weight.dtype == torch.bfloat16:
        bit_width = 16  # BF16
    elif layer['module'].weight.dtype == torch.float64:
        bit_width = 64  # FP64
    
    return bit_width


def get_relevant_bit_positions(bit_width):
    """
    Get a list of relevant bit positions to try flipping based on bit width.
    
    Args:
        bit_width: Number of bits per parameter
        
    Returns:
        bit_positions: List of bit positions to consider
    """
    # Prioritize bits based on their impact
    if bit_width == 32:  # FP32
        # For floating point, prioritize sign bit and exponent bits
        bit_positions = [31]  # Sign bit
        bit_positions.extend(range(30, 22, -1))  # Exponent bits
    elif bit_width == 16:  # FP16
        bit_positions = [15]  # Sign bit
        bit_positions.extend(range(14, 9, -1))  # Exponent bits
    elif bit_width == 64:  # FP64
        bit_positions = [63]  # Sign bit
        bit_positions.extend(range(62, 52, -1))  # Exponent bits
    else:  # Integer/quantized
        # Prioritize MSBs
        bit_positions = list(range(bit_width-1, max(bit_width-4, bit_width//2), -1))
    
    return bit_positions


def flip_bit(layer, param_idx, bit_pos):
    """
    Flip a specific bit in a model parameter.
    
    Args:
        layer: Layer information dictionary
        param_idx: Flattened parameter index
        bit_pos: Bit position to flip
        
    Returns:
        old_value: Original parameter value
        new_value: Value after bit flip
    """
    # Get module and parameter tensor
    module = layer['module']
    weight = module.weight
    
    # Convert flat index to tensor coordinates
    coords = np.unravel_index(param_idx, weight.shape)
    
    # Get current value
    value = weight.data[coords].item()
    old_value = value
    
    # Convert to binary representation and flip the bit
    if isinstance(value, float):
        # For floating point values, use bit-level manipulation
        if bit_pos >= 32:
            # Special case for double precision (64 bits)
            try:
                import struct
                # Convert double to its raw bit representation
                bits = struct.unpack('Q', struct.pack('d', value))[0]
                # Flip the specific bit
                bits ^= (1 << bit_pos)
                # Convert back to double
                new_value = struct.unpack('d', struct.pack('Q', bits))[0]
            except:
                # Fallback: just flip the sign for simplicity
                new_value = -value
        else:
            try:
                # Convert float to its bit representation
                int_repr = torch.tensor([value], dtype=torch.float32).view(torch.int32).item()
                # Flip the specific bit
                int_repr ^= (1 << bit_pos)
                # Convert back to float
                new_value = torch.tensor([int_repr], dtype=torch.int32).view(torch.float32).item()
            except:
                # Fallback: just flip the sign for simplicity
                new_value = -value
    else:
        # For integer values (quantized models)
        # Flip the bit directly
        new_value = value ^ (1 << bit_pos)
    
    # Update the parameter
    weight.data[coords] = new_value
    
    return old_value, new_value 