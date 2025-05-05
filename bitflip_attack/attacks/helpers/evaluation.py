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
            input_ids = None
            attention_mask = None
            labels = None

            # --- MODIFIED BATCH HANDLING ---
            if isinstance(batch, dict):
                # Directly extract from dict
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
                labels = batch.get('labels')
                # Create inputs dict for potential use in custom_forward_fn
                inputs_dict = {k: v for k, v in batch.items() if k != 'labels'} 
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # Handle tuple/list, potentially containing dicts
                inputs_data, labels_data = batch[0], batch[1]
                if isinstance(inputs_data, dict):
                     input_ids = inputs_data.get('input_ids')
                     attention_mask = inputs_data.get('attention_mask')
                     inputs_dict = inputs_data # Keep the dict
                else:
                     # Assume inputs_data is the tensor itself (e.g., image)
                     # How to get attention_mask? Assume None if not dict.
                     input_ids = inputs_data # Or pixel_values? Name depends on model 
                     attention_mask = None
                     inputs_dict = None # No dict available
                labels = labels_data # Assume second element is labels
            else:
                 print(" Skipping batch - unsupported format.")
                 continue
            
            # Check if we got necessary tensors
            if labels is None:
                 print(" Skipping batch - could not extract labels.")
                 continue
            # Need either input_ids+attn_mask OR the inputs_dict for model call
            if (input_ids is None or attention_mask is None) and inputs_dict is None:
                 print(" Skipping batch - could not extract sufficient input tensors.")
                 continue
                 
            # Move tensors to device
            labels = labels.to(device)
            if input_ids is not None: input_ids = input_ids.to(device)
            if attention_mask is not None: attention_mask = attention_mask.to(device)
            if inputs_dict: inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            # --- END MODIFIED BATCH HANDLING ---
            
            # Forward pass
            if custom_forward_fn is not None:
                # Pass the original batch structure as custom_forward_fn should handle it
                outputs = custom_forward_fn(model, batch)
            else:
                # Standard forward pass using extracted tensors
                # Assumes BERT-like model if input_ids/mask available
                if input_ids is not None and attention_mask is not None:
                     outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
                # Add alternative for other input types if needed (e.g., vision)
                # elif pixel_values is not None:
                #      outputs = model(pixel_values=pixel_values).logits 
                else:
                     print(" Skipping batch - cannot perform standard forward pass with extracted inputs.")
                     continue
            
            # Handle different output formats (Keep this as is)
            if isinstance(outputs, dict) and 'logits' in outputs:
                outputs = outputs['logits']
                
            # Calculate predictions
            _, predicted = outputs.max(1)
            
            # Calculate accuracy (use 'labels' extracted earlier)
            correct += (predicted == labels).sum().item()
            
            # Calculate attack success rate (use 'labels' extracted earlier)
            if attack_mode == 'targeted' and target_class is not None:
                attack_success += (predicted == target_class).sum().item()
            else:
                attack_success += (predicted != labels).sum().item()
            
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    asr = attack_success / total if total > 0 else 0.0
    
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
        # If accuracy is below threshold, penalize based on accuracy drop
        fitness = asr - (accuracy_threshold - accuracy)
    
    return fitness, accuracy, asr


def find_layer_by_name(layer_info, name):
    """Find layer in layer_info by name"""
    for layer in layer_info:
        if layer['name'] == name:
            return layer
    return None 