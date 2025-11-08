"""
Sensitivity analysis module for bit flip attacks

This module contains functions to analyze model layer sensitivity to bit flips.
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_sensitivity(model, layer, inputs, targets, 
                       use_gradient=True, use_magnitude=True, 
                       alpha=0.5, custom_forward_fn=None):
    """
    Compute sensitivity of parameters in a layer using the hybrid metric.
    
    Args:
        model: The model being analyzed
        layer: Layer information dictionary
        inputs: Batch of inputs for gradient computation
        targets: Corresponding labels
        use_gradient: Whether to use gradient information
        use_magnitude: Whether to use weight magnitude
        alpha: Weight between gradient (alpha) and magnitude (1-alpha)
        custom_forward_fn: Custom forward function for the model
        
    Returns:
        sensitivity: Tensor of sensitivity scores for each parameter
    """
    layer_module = layer['module']
    weight = layer_module.weight
    
    # Weight magnitude component
    weight_magnitude = torch.abs(weight.detach())
    
    # If using only magnitude or gradient isn't available
    if not use_gradient:
        return weight_magnitude
    
    # Compute gradient component by backpropagating
    model.zero_grad()
    
    # Forward pass with custom function if provided
    if custom_forward_fn is not None:
        # Reconstruct the dictionary batch expected by custom_forward_fn
        # `inputs` here is the dictionary {'input_ids':..., 'attention_mask':...}
        # `targets` is the labels tensor
        batch = inputs.copy() # Start with input_ids, attention_mask
        batch['labels'] = targets # Add labels back in
        outputs = custom_forward_fn(model, batch)
    else:
        # Standard model call (assuming it takes dict or specific args)
        outputs = model(**inputs)
    
    loss = F.cross_entropy(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Check if gradient exists
    if weight.grad is None:
        return weight_magnitude
    
    gradient_magnitude = torch.abs(weight.grad.detach())
    
    # Normalize weights and gradients to [0,1] range
    weight_min, weight_max = weight_magnitude.min(), weight_magnitude.max()
    if weight_min != weight_max:  # Avoid division by zero
        weight_norm = (weight_magnitude - weight_min) / (weight_max - weight_min)
    else:
        weight_norm = torch.zeros_like(weight_magnitude)
        
    grad_min, grad_max = gradient_magnitude.min(), gradient_magnitude.max()
    if grad_min != grad_max:  # Avoid division by zero
        grad_norm = (gradient_magnitude - grad_min) / (grad_max - grad_min)
    else:
        grad_norm = torch.zeros_like(gradient_magnitude)
    
    # Compute hybrid sensitivity
    if use_magnitude and use_gradient:
        sensitivity = alpha * grad_norm + (1 - alpha) * weight_norm
    elif use_gradient:
        sensitivity = grad_norm
    else:
        sensitivity = weight_norm
        
    return sensitivity


def rank_layers_by_sensitivity(model, dataset, layer_info, device, 
                              hybrid_sensitivity=True, alpha=0.5,
                              custom_forward_fn=None, n_samples=100):
    """
    Rank layers by their sensitivity to bit flips.
    
    Args:
        model: The model being analyzed
        dataset: Dataset for evaluating model performance
        layer_info: List of layer information dictionaries
        device: Device to run computations on
        hybrid_sensitivity: Whether to use hybrid sensitivity metric
        alpha: Weight for hybrid sensitivity metric
        custom_forward_fn: Custom forward function for the model
        n_samples: Number of samples to use for gradient computation
        
    Returns:
        sorted_layers: List of layer info sorted by sensitivity
    """
    # Get a batch of data for gradient computation
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=min(n_samples, len(dataset)), shuffle=True
    )
    
    batch = next(iter(dataloader))
    
    # Handle different batch formats
    if isinstance(batch, dict):
        # Dictionary batch (common in transformers)
        # Move all tensors to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        targets = batch['labels']
        # Keep inputs as dict with input_ids and attention_mask for model forward
        inputs = {'input_ids': batch['input_ids']}
        if 'attention_mask' in batch:
            inputs['attention_mask'] = batch['attention_mask']
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        # Tuple/list batch (traditional format)
        inputs, targets = batch[0], batch[1]
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)
        targets = targets.to(device)
    else:
        raise ValueError("Unable to extract inputs and targets from dataset")
    
    # Put model in evaluation mode but enable grad tracking
    model.eval()
    model.zero_grad()
    for param in model.parameters():
        param.requires_grad = True
            
    layer_losses = []
    
    # For each layer, compute sensitivity and induce perturbations
    for idx, layer in enumerate(layer_info):
        # Skip layers without weights
        if not hasattr(layer['module'], 'weight') or layer['module'].weight is None:
            continue
            
        # Compute sensitivity score
        try:
            sensitivity = compute_sensitivity(
                model, layer, inputs, targets, 
                use_gradient=hybrid_sensitivity, 
                use_magnitude=True,
                alpha=alpha,
                custom_forward_fn=custom_forward_fn
            )
            
            # Sample bits to flip based on sensitivity
            k = int(layer['n_params'] * 0.001)  # Flip 0.1% of bits for test
            k = max(1, min(k, 100))  # Cap between 1 and 100 bits
            
            # Get top-k indices by sensitivity
            _, indices = torch.topk(sensitivity.view(-1), k)
            
            # Store original weights
            orig_weight = layer['module'].weight.data.clone()
            
            # Apply bit flips to MSB (most significant bit) of selected weights
            for idx_bit in indices:
                # Convert flat index to tensor coordinates
                coords = np.unravel_index(idx_bit.item(), layer['shape'])
                coords = tuple(coords)
                
                # Bit flip at MSB (we'll flip the sign bit for most impact)
                value = layer['module'].weight.data[coords].item()
                # Simple flip for testing: flip the sign to maximize impact
                layer['module'].weight.data[coords] = -value
            
            # Compute loss after perturbation
            model.zero_grad()
            if custom_forward_fn is not None:
                # Reconstruct the dictionary batch expected by custom_forward_fn
                batch = inputs.copy() # Start with input_ids, attention_mask
                batch['labels'] = targets # Add labels back in
                outputs = custom_forward_fn(model, batch)
            else:
                # Standard model call (assuming it takes dict or specific args)
                outputs = model(**inputs)
                
            loss = F.cross_entropy(outputs, targets)
            
            # Restore original weights
            layer['module'].weight.data.copy_(orig_weight)
            
            # Store layer info with sensitivity score
            layer_info_with_loss = {
                'index': idx,
                'name': layer['name'],
                'type': layer['type'],
                'n_params': layer['n_params'],
                'loss': loss.item(),
                'sensitivity': sensitivity.mean().item()
            }
            layer_losses.append(layer_info_with_loss)
        except Exception as e:
            print(f"Error analyzing layer {layer['name']}: {e}")
    
    # Sort layers by loss (higher loss = more sensitive)
    sorted_layers = sorted(layer_losses, key=lambda x: x['loss'], reverse=True)
    
    # Reset model gradients
    model.zero_grad()
    
    return sorted_layers 