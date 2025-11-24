"""
Evaluation module for bit flip attacks

This module contains functions to evaluate model performance and attack success rates.
"""
import torch


def evaluate_model_performance(model, dataset, target_class=None, 
                               attack_mode='targeted', device='cuda',
                               custom_forward_fn=None, batch_size=32):
    """
    Evaluate model performance and attack success rate.
    
    Args:
        model: The model to evaluate
        dataset: Dataset for evaluation
        target_class: Target class for targeted attacks (None for untargeted)
        attack_mode: 'targeted' or 'untargeted'
        device: Device to run evaluation on
        custom_forward_fn: Custom forward function for the model
        batch_size: Batch size for evaluation
        
    Returns:
        accuracy: Model accuracy on clean test set
        asr: Attack success rate
    """
    model.eval()
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    correct = 0
    attack_success = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Ensure batch is a dictionary
            if not isinstance(batch, dict):
                # If somehow it's not a dict, try to handle tuple/list or raise error
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                     print("Warning: evaluate_model_performance received tuple/list batch, expected dict. Attempting conversion.")
                     batch = {'image': batch[0], 'label': batch[1]}
                else:
                     raise TypeError(f"evaluate_model_performance expects batch to be a dict, but received {type(batch)}")

            # Extract tensors from dictionary and move to device
            # Support both vision models (image/label) and NLP models (input_ids/labels)
            if 'image' in batch:
                # Vision model
                inputs = batch['image'].to(device)
                targets = batch['label'].to(device)
                model_inputs = inputs
            elif 'input_ids' in batch:
                # NLP model
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                raise ValueError(f"Batch must contain either 'image' or 'input_ids', got keys: {batch.keys()}")
            
            # Forward pass with custom function if provided
            if custom_forward_fn is not None:
                # Pass the dictionary batch directly to the custom function
                # as it's designed to handle it (like in umup_attack_example.py)
                outputs = custom_forward_fn(model, batch) 
            else:
                 # Standard model call
                 if isinstance(model_inputs, dict):
                     # NLP model with dict inputs
                     outputs = model(**model_inputs)
                 else:
                     # Vision model with tensor input
                     outputs = model(model_inputs)
            
            # Handle different output formats
            if isinstance(outputs, dict) and 'logits' in outputs:
                outputs = outputs['logits']
                
            # Calculate predictions
            _, predicted = outputs.max(1)
            
            # Calculate accuracy
            correct += (predicted == targets).sum().item()
            
            # Calculate attack success rate
            if attack_mode == 'targeted' and target_class is not None:
                # For targeted attacks: success if model predicts target class
                attack_success += (predicted == target_class).sum().item()
            else:
                # For untargeted attacks: success if model prediction is wrong
                attack_success += (predicted != targets).sum().item()
            
            total += targets.size(0)
    
    accuracy = correct / total
    asr = attack_success / total
    
    return accuracy, asr


def evaluate_individual_fitness(model, dataset, individual, candidates, layer_info,
                               target_class, attack_mode, accuracy_threshold,
                               device='cuda', custom_forward_fn=None):
    """
    Evaluate fitness of an individual in the genetic algorithm.
    
    Args:
        model: The model to evaluate
        dataset: Dataset for evaluation
        individual: List of indices into candidates list
        candidates: List of bit flip candidates
        layer_info: List of layer information dictionaries
        target_class: Target class for targeted attacks
        attack_mode: 'targeted' or 'untargeted'
        accuracy_threshold: Minimum accuracy to maintain
        device: Device to run evaluation on
        custom_forward_fn: Custom forward function for the model
        
    Returns:
        fitness: Fitness score for individual
        accuracy: Model accuracy
        asr: Attack success rate
    """
    # Apply bit flips for this individual
    for idx in individual:
        candidate = candidates[idx]
        layer_idx = candidate['layer_idx']
        layer = layer_info[layer_idx] if layer_idx >= 0 else find_layer_by_name(layer_info, candidate['layer_name'])
        param_idx = candidate['parameter_idx']
        bit_pos = candidate['bit_position']
        
        from bitflip_attack.attacks.helpers.bit_manipulation import flip_bit
        flip_bit(layer, param_idx, bit_pos)
    
    # Evaluate model
    accuracy, asr = evaluate_model_performance(
        model, dataset, target_class, attack_mode, device, custom_forward_fn
    )
    
    # Compute fitness: maximize ASR while keeping accuracy above threshold
    if accuracy >= accuracy_threshold:
        # If accuracy is acceptable, focus on maximizing ASR
        fitness = asr
    else:
        # If accuracy is below threshold, penalize heavily (multiply by 5 to discourage)
        penalty = 5.0 * (accuracy_threshold - accuracy)
        fitness = asr - penalty
        # Ensure fitness doesn't go too negative
        fitness = max(fitness, -1.0)
    
    return fitness, accuracy, asr


def find_layer_by_name(layer_info, name):
    """Find layer in layer_info by name"""
    for layer in layer_info:
        if layer['name'] == name:
            return layer
    return None 