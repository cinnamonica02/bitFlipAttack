#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn.functional as F # For loss functions
import torch.optim as optim    # For optimizers
import numpy as np
import os
import time
from datetime import datetime
import copy # Needed for deepcopy
import random
import traceback # <-- Import traceback module

# Import necessary helper functions (potentially reusing/adapting)
from bitflip_attack.attacks.helpers.bit_manipulation import flip_bit
from bitflip_attack.attacks.helpers.evaluation import evaluate_model_performance
from bitflip_attack.attacks.helpers.sensitivity import compute_sensitivity
from torch.utils.data import DataLoader, Subset, ConcatDataset # Added dataset utils
from torch.utils.data import TensorDataset # For creating Dcert
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Temporary imports for __main__ block (adjust later)
from transformers import BertTokenizer, BertForSequenceClassification
# Assuming PIIDataset and load_or_create_dataset exist similarly to the other example
# Need to adjust path if groan_bitflip_attack is run directly
import sys
# Add parent example directory to path to find PIIDataset etc.
# This is fragile and should be fixed with better project structure/imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
try:
    # Attempt to import from the example script's location
    from umup_attack_example import PIIDataset, load_or_create_dataset 
except ImportError as e:
    print(f"Warning: Could not import from umup_attack_example ({e}). __main__ block might fail.")
    PIIDataset = None
    load_or_create_dataset = None

# TODO: Define the GroanAttack class or core functions

class GroanAttack:
    def __init__(self, model, dataset, device='cuda', custom_forward_fn=None):
        """Initialize the Groan attack.
        Args:
            model: The target model (encoder part accessible)
            dataset: Dataset for substitute model training/evaluation
            device: Device to run on
            custom_forward_fn: Custom forward function
        """
        self.model = model # Assume this is the full model for now
        self.dataset = dataset
        self.device = device
        self.custom_forward_fn = custom_forward_fn
        self.substitute_model = None
        self.flipper = None # Placeholder for bit flipping mechanism
        self.layer_info = self._get_layer_info() # Get layer info on init

        print(f"GroanAttack initialized with {len(self.layer_info)} layers identified.")

    def _get_layer_info(self):
        """Analyze model structure and identify layers with parameters.
           Adapted from BitFlipAttack class.
        Returns:
            List of dictionaries containing layer information
        """
        layer_info = []
        for name, module in self.model.named_modules():
            # Skip containers and non-parameter layers
            if len(list(module.children())) > 0:
                continue
                
            # Focus on layers with weights (can be extended to bias later if needed)
            if hasattr(module, 'weight') and module.weight is not None:
                n_params = module.weight.numel()
                if n_params == 0:
                    continue
                
                layer_info.append({
                    'name': name,
                    'module': module,
                    'type': module.__class__.__name__,
                    'n_params': n_params,
                    'shape': module.weight.shape,
                    'requires_grad': module.weight.requires_grad,
                    # Add quantization info check if needed later
                    # 'is_quantized': ...
                })
        
        return layer_info

    # --- Helper Methods for Knowledge Discovery ---

    def _train_substitute(self, dataset, epochs=1, lr=5e-5):
        """Basic training/fine-tuning for the substitute model."""
        if self.substitute_model is None:
            print("  Warning: Substitute model not initialized. Skipping training.")
            return
        if len(dataset) == 0:
            print("  Warning: Dataset is empty. Skipping training.")
            return
            
        print(f"  Training substitute on {len(dataset)} samples for {epochs} epoch(s)...")
        
        self.substitute_model.train() # Set model to training mode
        optimizer = AdamW(self.substitute_model.parameters(), lr=lr)
        loss_fn = CrossEntropyLoss()
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # Use a reasonable batch size

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Handle data format (assuming dict for simplicity now)
                if not isinstance(batch, dict) or 'input_ids' not in batch or 'attention_mask' not in batch or 'labels' not in batch:
                    # print(" Skipping incompatible batch format in training.")
                    continue # Skip incompatible batches
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.substitute_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"    Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
        
        self.substitute_model.eval() # Set back to eval mode
        print("  Substitute training finished.")

    def _calculate_uncertainty(self, model, dataloader):
        """Calculate prediction uncertainty (e.g., negative entropy) for samples."""
        model.eval()
        uncertainties = []
        indices = []
        print(f"  Calculating uncertainty for {len(dataloader.dataset)} samples...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader): # Assuming dataloader yields data suitable for model
                 # Handle dict or tuple batch format (adapt from custom_forward_fn)
                input_dict = None
                if isinstance(batch, tuple) and len(batch) == 2: input_dict = batch[0]
                elif isinstance(batch, dict): input_dict = batch
                else: continue # Skip incompatible batches silently for now

                if not (isinstance(input_dict, dict) and 'input_ids' in input_dict and 'attention_mask' in input_dict):
                    continue # Skip if format is wrong

                input_ids = input_dict['input_ids'].to(self.device)
                attention_mask = input_dict['attention_mask'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                # Entropy: -sum(p * log(p)) -- Higher value means more uncertainty
                # We can store entropy directly or use negative entropy if sorting for least uncertain later
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1) # Add epsilon for stability
                uncertainties.extend(entropy.cpu().tolist())
                # Need to track original indices if using subsets/shuffled dataloaders
                # Simplification: Assume dataloader iterates in order for now
                batch_indices = list(range(i * dataloader.batch_size, 
                                           min((i + 1) * dataloader.batch_size, len(dataloader.dataset))))
                indices.extend(batch_indices) 

        # Return indices sorted by uncertainty (most uncertain first)
        sorted_indices = [idx for _, idx in sorted(zip(uncertainties, indices), reverse=True)]
        return sorted_indices, uncertainties # Also return raw uncertainties if needed

    def _select_inputs_by_uncertainty(self, model, dataset, num_samples, most_uncertain=True):
        """Select indices of most/least uncertain samples."""
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False) # Use larger batch size for eval
        sorted_indices, _ = self._calculate_uncertainty(model, dataloader)
        
        if most_uncertain:
            selected_indices = sorted_indices[:num_samples]
        else: # Least uncertain (most certain)
            selected_indices = sorted_indices[-num_samples:]
            
        print(f"  Selected {len(selected_indices)} {'most' if most_uncertain else 'least'} uncertain samples.")
        # Return a subset of the original dataset corresponding to these indices
        return torch.utils.data.Subset(dataset, selected_indices)

    def _generate_trigger(self, model, dataset, target_class, epochs=5):
        """Generates an initial trigger by optimizing on the provided model and dataset."""
        print("  Generating initial trigger...")
        # This dataset should ideally be representative, e.g., D
        # Ensure dataset is not empty
        if len(dataset) == 0:
             print("  Warning: Dataset for trigger generation is empty. Skipping.")
             self.current_trigger_pattern = None
             return self.apply_trigger # Return dummy applicator

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        # Start with a None trigger, _update_trigger will initialize if None
        optimized_trigger = self._update_trigger(model, None, dataloader, target_class, epochs=epochs)
        # _update_trigger now returns the trigger pattern, not a function
        self.current_trigger_pattern = optimized_trigger # Store the pattern
        print("  Initial trigger generated.")
        # Return a function that applies the stored trigger pattern
        return self.apply_trigger # Return the instance method

    def apply_trigger(self, batch_data):
        """Applies the current trigger pattern to a batch of data (image focus)."""
        if self.current_trigger_pattern is None:
            # print("    Trigger pattern is None, returning original batch.")
            return batch_data
            
        # Try to find image tensor (assuming NCHW format)
        images = None
        original_input_structure = batch_data # Keep track for returning same structure
        
        if isinstance(batch_data, dict):
            if 'pixel_values' in batch_data: images = batch_data['pixel_values']
            elif 'image' in batch_data: images = batch_data['image']
            # Add other potential keys
        elif isinstance(batch_data, torch.Tensor) and len(batch_data.shape) == 4:
            images = batch_data
        elif isinstance(batch_data, (list, tuple)) and len(batch_data) > 0 and isinstance(batch_data[0], torch.Tensor) and len(batch_data[0].shape) == 4:
            images = batch_data[0] # Assume first element is image tensor
            
        if images is not None:
            try:
                triggered_images = images.clone().to(self.device) # Work on a copy on correct device
                trigger = self.current_trigger_pattern.to(triggered_images.device) # Ensure trigger is on same device
                p_c, p_h, p_w = trigger.shape
                img_c, img_h, img_w = triggered_images.shape[1:]
                
                if p_c != img_c:
                     print(f"    Warning: Trigger channels ({p_c}) != Image channels ({img_c}). Skipping trigger application.")
                     return original_input_structure

                # Apply to bottom-right corner
                start_h = img_h - p_h
                start_w = img_w - p_w
                if start_h >= 0 and start_w >= 0:
                    triggered_images[:, :, start_h:, start_w:] = triggered_images[:, :, start_h:, start_w:] + trigger
                    # Clamp based on original image range (estimate)
                    min_val, max_val = images.min().item(), images.max().item()
                    triggered_images = torch.clamp(triggered_images, min_val, max_val)
                    
                    # Put modified tensor back into original structure
                    if isinstance(original_input_structure, dict):
                         # Find the key used
                         key_used = 'pixel_values' if 'pixel_values' in original_input_structure else 'image' 
                         modified_batch = original_input_structure.copy()
                         modified_batch[key_used] = triggered_images
                         return modified_batch
                    elif isinstance(original_input_structure, torch.Tensor):
                         return triggered_images
                    elif isinstance(original_input_structure, (list, tuple)):
                         # Assume structure (image_tensor, other_stuff...)
                         modified_batch_list = list(original_input_structure)
                         modified_batch_list[0] = triggered_images
                         return tuple(modified_batch_list) if isinstance(original_input_structure, tuple) else modified_batch_list
                else:
                    print("    Warning: Trigger size larger than image. Skipping trigger application.")
                    return original_input_structure
            except Exception as e:
                 print(f"    Error applying trigger: {e}")
                 return original_input_structure # Return original on error
        else:
             # print("    Could not identify image tensor in batch for trigger application.")
             return original_input_structure # Return original if no image tensor found

    # --- Main Knowledge Discovery Method ---

    def knowledge_discovery(self, target_class, query_budget, queries_per_iteration=1000):
        """Implement Algorithm 1: Knowledge Discovery.
           Train/refine substitute model and prepare datasets Dl, Duncert, Dcert.
        Args:
            target_class: The target class label for the Trojan attack.
            query_budget: Maximum number of queries allowed to the target model (Q).
            queries_per_iteration: Number of queries per refinement iteration (q).
        """
        print(f"Starting Knowledge Discovery (Target: {target_class}, Budget: {query_budget}, Per Iter: {queries_per_iteration})...")
        
        q = queries_per_iteration
        Q = query_budget
        Nq = 0 # Number of queries made so far

        # Assume self.dataset is the initial pool of unlabeled data available
        initial_pool_indices = list(range(len(self.dataset)))
        random.shuffle(initial_pool_indices)
        
        # 1. Initial random sampling (Line 1-2)
        print("Step 1: Initial Random Sampling...")
        initial_query_indices = initial_pool_indices[:q]
        initial_query_subset = torch.utils.data.Subset(self.dataset, initial_query_indices)
        # TODO: Query the *actual* target model 'f' here. Using self.model as placeholder.
        Dr_list = [] 
        print(f"  Querying target model with {len(initial_query_subset)} samples...")
        temp_loader = torch.utils.data.DataLoader(initial_query_subset, batch_size=32)
        for batch in temp_loader: # Simulate getting labels
             input_dict = None
             if isinstance(batch, tuple): input_dict = batch[0]; label_tensor = batch[1] # Use existing label if available? No, query f
             elif isinstance(batch, dict): input_dict = batch; label_tensor = batch['labels'] # Use existing label? No, query f
             else: continue

             # --- Simulate querying f ---
             # In real scenario, send input_dict (or underlying data) to f, get label
             simulated_labels = label_tensor # *** REPLACE WITH ACTUAL QUERY TO f ***
             # -------------------------
             
             # Store data with label from f 
             # Need to handle batch collation carefully if not using a standard Dataset structure
             # For now, assume we collect samples like {'input_ids':..., 'attention_mask':..., 'labels': actual_label_from_f}
             # This part needs a proper Dataset implementation for Dr
        Dr = initial_query_subset # Placeholder: Use the subset itself with its original labels for now
        Nq += len(Dr)
        print(f"  Collected Dr with {len(Dr)} samples. Total queries: {Nq}")

        # Initialize substitute model (Line 3)
        print("Step 2: Initial Substitute Training...")
        self.substitute_model = copy.deepcopy(self.model) # Start with a copy of the accessible model
        self._train_substitute(Dr, epochs=3) # Train initial substitute

        # Prepare overall dataset D for iterations
        D = Dr # Start with the randomly labeled set

        # 2. Iterative refinement loop (Line 5-11)
        print("Step 3: Iterative Substitute Refinement...")
        Dclean_uncert_list = [] # Keep track of uncertain sets per iteration
        while Nq < Q:
            print(f"  Iteration with Nq = {Nq} / {Q}")
            if len(D) == 0:
                 print("  Warning: Dataset D is empty, cannot select uncertain samples. Stopping refinement.")
                 break
            # Select uncertain inputs from D (Line 6)
            uncertain_subset = self._select_inputs_by_uncertainty(self.substitute_model, D, q, most_uncertain=True)
            
            # Query target model 'f' for labels (Line 7)
            print(f"  Querying target model with {len(uncertain_subset)} uncertain samples...")
            # *** Simulate query process - NEEDS REAL IMPLEMENTATION ***
            Dclean_uncert_current_iter = uncertain_subset # Placeholder: Use the subset with existing labels
            Nq += len(Dclean_uncert_current_iter)
            print(f"  Collected Dclean_uncert with {len(Dclean_uncert_current_iter)} samples. Total queries: {Nq}")
            if Nq >= Q: # Trim if over budget
                 excess = Nq - Q
                 if excess < len(Dclean_uncert_current_iter):
                      Dclean_uncert_current_iter = Subset(Dclean_uncert_current_iter, list(range(len(Dclean_uncert_current_iter) - excess)))
                 else: # Should not happen if q <= Q-Nq_before
                      Dclean_uncert_current_iter = Subset(Dclean_uncert_current_iter, []) # Empty subset
            
            Dclean_uncert_list.append(Dclean_uncert_current_iter) # Store this iteration's set

            # Update substitute (Line 8)
            if len(Dclean_uncert_current_iter) > 0:
                 self._train_substitute(Dclean_uncert_current_iter, epochs=1)
            
            # Update D (Line 9)
            if len(Dclean_uncert_current_iter) > 0:
                D = ConcatDataset([D, Dclean_uncert_current_iter])
            print(f"  Dataset D size: {len(D)}")

            if Nq >= Q: break
        
        print("Step 4: Trigger-based Substitute Refinement...")
        if len(D) == 0:
            print("  Warning: Dataset D is empty, cannot generate trigger. Skipping trigger refinement.")
            trigger_func = lambda x: x # Dummy trigger function
            Dtrigger_uncert = Subset(self.dataset, []) # Empty dataset
        else:
            # Get trigger (Line 12)
            trigger_func = self._generate_trigger(self.substitute_model, D, target_class) # Generate initial trigger

            # Apply trigger (conceptually) and select uncertain inputs (Line 13)
            # Placeholder: Select uncertain inputs from D without trigger for now
            trigger_uncertain_subset = self._select_inputs_by_uncertainty(self.substitute_model, D, q, most_uncertain=True)
            
            # Query target model 'f' (Line 14)
            print(f"  Querying target model with {len(trigger_uncertain_subset)} trigger-uncertain samples...")
            # *** Simulate query process - NEEDS REAL IMPLEMENTATION ***
            Dtrigger_uncert = trigger_uncertain_subset # Placeholder
            Nq += len(Dtrigger_uncert)
            if Nq > Q: # Trim
                 excess = Nq - Q
                 if excess < len(Dtrigger_uncert):
                     Dtrigger_uncert = Subset(Dtrigger_uncert, list(range(len(Dtrigger_uncert) - excess)))
                 else:
                     Dtrigger_uncert = Subset(Dtrigger_uncert, [])
            print(f"  Collected Dtrigger_uncert with {len(Dtrigger_uncert)} samples. Total queries: {Nq}")

            # Update substitute (Line 15)
            if len(Dtrigger_uncert) > 0:
                self._train_substitute(Dtrigger_uncert, epochs=1)

        print("Step 5: Final Dataset Preparation...")
        # Prepare Duncert (Line 16)
        # Use uncertain sets from the last clean iteration and the trigger iteration
        last_clean_uncert = Dclean_uncert_list[-1] if Dclean_uncert_list else Subset(self.dataset, [])
        Duncert = ConcatDataset([last_clean_uncert, Dtrigger_uncert])
        print(f"  Prepared Duncert with {len(Duncert)} samples.")

        # Prepare Dcert (Line 17-18)
        Dcert = Subset(self.dataset, []) # Initialize as empty
        if len(D) > 0 or len(Dtrigger_uncert) > 0:
            certainty_source_dataset = ConcatDataset([D, Dtrigger_uncert]) if len(Dtrigger_uncert) > 0 else D
            if len(certainty_source_dataset) > 0:
                num_cert_samples = min(q * 2, len(certainty_source_dataset))
                Dcert_subset = self._select_inputs_by_uncertainty(self.substitute_model, certainty_source_dataset, num_cert_samples, most_uncertain=False)
                
                # Label Dcert using the *substitute* model
                print(f"  Labeling {len(Dcert_subset)} certain samples using substitute model...")
                cert_inputs = []
                cert_masks = []
                cert_substitute_labels = []
                cert_loader = DataLoader(Dcert_subset, batch_size=32)
                self.substitute_model.eval()
                with torch.no_grad():
                    for batch in cert_loader:
                        # Handle data format
                        input_dict = None
                        if isinstance(batch, tuple): input_dict = batch[0]
                        elif isinstance(batch, dict): input_dict = batch
                        else: continue
                        
                        if not (isinstance(input_dict, dict) and 'input_ids' in input_dict and 'attention_mask' in input_dict):
                            continue
                            
                        input_ids = input_dict['input_ids'].to(self.device)
                        attention_mask = input_dict['attention_mask'].to(self.device)
                        outputs = self.substitute_model(input_ids=input_ids, attention_mask=attention_mask)
                        preds = torch.argmax(outputs.logits, dim=1)
                        
                        cert_inputs.append(input_ids.cpu())
                        cert_masks.append(attention_mask.cpu())
                        cert_substitute_labels.append(preds.cpu())
                
                if cert_inputs:
                    all_inputs = torch.cat(cert_inputs, dim=0)
                    all_masks = torch.cat(cert_masks, dim=0)
                    all_labels = torch.cat(cert_substitute_labels, dim=0)
                    # Create a standard TensorDataset for Dcert
                    # Note: This loses original dict structure if it had other keys
                    Dcert = TensorDataset(all_inputs, all_masks, all_labels)
                    print(f"  Prepared Dcert with {len(Dcert)} samples (labeled by substitute).")
                else:
                    print("  Warning: No valid data found in Dcert_subset to label.")
            else:
                 print("  Warning: Certainty source dataset is empty. Cannot create Dcert.")
        else:
             print("  Warning: Dataset D and Dtrigger_uncert are empty. Cannot create Dcert.")

        # Prepare Dl (Line 19)
        self.Dl = Duncert # Use uncertain set for bit identification for now
        print(f"  Prepared Dl (using Duncert) with {len(self.Dl)} samples.")

        # Keep Duncert and Dcert separately if needed by identify_bits
        self.Duncert = Duncert
        self.Dcert = Dcert
        
        print(f"Knowledge Discovery Complete. Total Queries: {Nq}. Final Substitute Trained.")
        # Line 20: Return ˜f, Dl (self.substitute_model and self.Dl are now set)

    # --- Helper Methods for Bit Identification ---

    def _update_trigger(self, model_state, current_trigger_pattern, dataloader, target_class, epochs=5, lr=0.01):
        """Simplified T-step: Optimize a trigger pattern to maximize target class probability."""
        print("    T-Step: Updating trigger (Simplified Gradient Ascent)...")
        model_state.eval() # Ensure model is in eval mode

        # Initialize trigger if None (e.g., small random patch)
        # Assuming image data for now, needs adaptation based on actual data type
        # Placeholder: guess input shape from dataloader if possible
        if dataloader is None or len(dataloader.dataset) == 0:
             print("    Warning: Dataloader is empty or None, cannot optimize trigger.")
             return current_trigger_pattern

        try:
            sample_batch = next(iter(dataloader))
            # Find likely input tensor based on common keys or structure
            # This part needs to be robust based on expected dataset structure
            input_tensor_shape = None
            if isinstance(sample_batch, dict):
                 if 'pixel_values' in sample_batch: input_tensor_shape = sample_batch['pixel_values'].shape
                 # Add checks for other keys like 'image', 'input' etc.
                 elif 'input_ids' in sample_batch: pass # Cannot easily determine shape for NLP trigger patch
            elif isinstance(sample_batch, (list, tuple)) and len(sample_batch) > 0 and isinstance(sample_batch[0], torch.Tensor):
                 input_tensor_shape = sample_batch[0].shape
            elif isinstance(sample_batch, torch.Tensor):
                 input_tensor_shape = sample_batch.shape


            if input_tensor_shape is not None and len(input_tensor_shape) == 4: # e.g., (N, C, H, W)
                 C, H, W = input_tensor_shape[1:]
                 patch_size = min(H // 4, W // 4, 10) # Example patch size
                 if current_trigger_pattern is None:
                     print(f"      Initializing trigger pattern ({C}x{patch_size}x{patch_size})...")
                     trigger_pattern = torch.rand((C, patch_size, patch_size), device=self.device, requires_grad=True)
                 else:
                     trigger_pattern = current_trigger_pattern.detach().clone().to(self.device).requires_grad_(True)
                 
                 optimizer = optim.Adam([trigger_pattern], lr=lr)
                 print(f"      Optimizing trigger for {epochs} epochs...")

                 for epoch in range(epochs):
                     total_loss = 0
                     num_batches = 0
                     for batch in dataloader:
                         # --- Apply Trigger (Needs data-specific implementation) ---
                         # This is a placeholder. Real application depends on data.
                         # Assume batch is modified IN PLACE by apply_trigger if needed,
                         # OR apply trigger here before passing to model.
                         # triggered_batch = self.apply_trigger(batch) # Example
                         # Let's assume for now the model input part handles it conceptually

                         # Get inputs (handle dict/tuple) - needs refinement
                         input_ids, attention_mask = None, None # Default for NLP
                         pixel_values = None # Default for Vision
                         
                         if isinstance(batch, dict):
                             input_ids = batch.get('input_ids')
                             attention_mask = batch.get('attention_mask')
                             pixel_values = batch.get('pixel_values')
                             # Add more potential keys
                         elif isinstance(batch, (list, tuple)):
                             # Assume standard structure if tuple/list
                             if len(batch) >= 2:
                                 if isinstance(batch[0], dict): # Like (input_dict, label)
                                     input_ids = batch[0].get('input_ids')
                                     attention_mask = batch[0].get('attention_mask')
                                     pixel_values = batch[0].get('pixel_values')
                                 elif isinstance(batch[0], torch.Tensor): # Like (tensor_input, label)
                                     # How to know if NLP or Vision? Assume vision if 4D?
                                     if len(batch[0].shape) == 4: pixel_values = batch[0]
                                     # Needs better way to determine input type

                         # Determine model input based on what was found
                         if pixel_values is not None:
                              model_input_args = {'pixel_values': pixel_values.to(self.device)}
                         elif input_ids is not None and attention_mask is not None:
                              model_input_args = {'input_ids': input_ids.to(self.device),
                                                 'attention_mask': attention_mask.to(self.device)}
                         else:
                              print("     Skipping batch - could not determine valid model inputs.")
                              continue

                         # --- Apply Additive Trigger Patch (Example for Vision) ---
                         if pixel_values is not None and trigger_pattern is not None:
                              images = model_input_args['pixel_values']
                              p_size = trigger_pattern.shape[-1]
                              start_h = images.shape[-2] - p_size
                              start_w = images.shape[-1] - p_size
                              if start_h >= 0 and start_w >= 0:
                                   images[:, :, start_h:, start_w:] = images[:, :, start_h:, start_w:] + trigger_pattern
                                   # Clamp to a reasonable range, assuming input is e.g. [-1, 1] or [0, 1]
                                   images = torch.clamp(images, images.min().item(), images.max().item()) # Simple clamp
                                   model_input_args['pixel_values'] = images
                         # -----------------------------------------------------------

                         optimizer.zero_grad()

                         # Use custom forward fn if provided? The paper implies querying the *whole* model
                         # but optimization happens on the *substitute*. Assume substitute call.
                         outputs = model_state(**model_input_args).logits


                         # Loss: Negative log likelihood of the target class
                         target_labels = torch.full((outputs.size(0),), target_class, dtype=torch.long, device=self.device)
                         loss = -F.cross_entropy(outputs, target_labels) # Minimize negative CE

                         loss.backward()
                         # Gradient clipping can be useful for trigger optimization
                         # torch.nn.utils.clip_grad_norm_([trigger_pattern], max_norm=1.0)
                         optimizer.step()

                         # Optional: Clamp trigger pattern values itself after step
                         # trigger_pattern.data.clamp_(0, 1) # Example for normalized image patch values

                         total_loss += loss.item()
                         num_batches += 1

                     avg_loss = total_loss / num_batches if num_batches > 0 else 0
                     print(f"      Epoch {epoch+1}/{epochs}, Avg Loss (Neg CE): {avg_loss:.4f}")

                 print("    T-Step: Trigger optimization finished.")
                 return trigger_pattern.detach() # Return the optimized pattern
            
            else:
                 print("    Warning: Could not determine input shape for trigger initialization.")
                 return current_trigger_pattern # Return unchanged

        except StopIteration:
             print("    Warning: Dataloader is empty, cannot optimize trigger.")
             return current_trigger_pattern # Return unchanged
        except Exception as e:
             print(f"    Error during trigger optimization: {e}")
             traceback.print_exc() # Print traceback for debugging
             return current_trigger_pattern # Return unchanged

    # --- MODIFIED _identify_vuln_bit (B-Step) --- 
    def _identify_vuln_bit(self, model_state, trigger_pattern, current_bits_info, target_class, acc_threshold, num_bits_to_sample=50):
        """Simplified B-step: Find the best bit to flip from a random sample.

        Evaluates a sample of potential bit flips and chooses the one that maximizes
        ASR while maintaining accuracy above the threshold.
        """
        print("    B-Step: Identifying vulnerable bit from sample...")
        if not self.layer_info:
            print("    Warning: No layer info found, cannot select bit.")
            return None

        best_bit_info = None
        best_asr = -1.0 # Track the best ASR found so far that meets ACC threshold
        current_acc, current_asr = self._evaluate_attack(model_state, trigger_pattern, self.Dl, target_class)

        potential_bits_to_try = []
        # Sample potential bits randomly from available layers
        # TODO: Integrate actual hardware constraints (flippable bits)
        for _ in range(num_bits_to_sample):
            # Choose a random layer (could weight by sensitivity later)
            layer_info = random.choice(self.layer_info)
            if layer_info['n_params'] == 0: continue
            param_idx = random.randint(0, layer_info['n_params'] - 1)
            
            # Determine bit width (rough guess)
            dtype = layer_info['module'].weight.dtype
            if dtype == torch.float32: bit_width = 32
            elif dtype == torch.float16: bit_width = 16
            elif dtype == torch.int8: bit_width = 8
            else: bit_width = 32
            bit_pos = random.randint(0, bit_width - 1)
            
            potential_bits_to_try.append({
                'layer_name': layer_info['name'],
                'param_idx': param_idx,
                'bit_pos': bit_pos,
                'layer_module': layer_info['module'] # Original module reference
            })

        print(f"    Evaluating {len(potential_bits_to_try)} potential bit flips...")
        for bit_info in potential_bits_to_try:
            # Create a copy of the current state to simulate the flip
            temp_model_state = copy.deepcopy(model_state)
            
            # Find the module in the *copy* using the name
            module_to_flip = None
            try:
                 module_to_flip = temp_model_state.get_submodule(bit_info['layer_name'])
            except AttributeError:
                 print(f"    Warning: Could not find submodule {bit_info['layer_name']} in temp model state.")
                 continue

            # Find the layer info dict for the original module
            original_layer_info = next((li for li in self.layer_info if li['name'] == bit_info['layer_name']), None)
            if not original_layer_info:
                 print(f"    Warning: Could not find original layer info for {bit_info['layer_name']}.")
                 continue
                 
            # Create a temp layer dict pointing to the module in the *copied* state
            temp_layer_dict = original_layer_info.copy()
            temp_layer_dict['module'] = module_to_flip
            
            # Simulate the flip
            try:
                flip_bit(temp_layer_dict, bit_info['param_idx'], bit_info['bit_pos'])
            except Exception as e:
                # print(f"    Error flipping bit {bit_info} during B-step simulation: {e}")
                continue # Skip if flip fails

            # Evaluate the temporary state
            eval_acc, eval_asr = self._evaluate_attack(temp_model_state, trigger_pattern, self.Dl, target_class)

            # Check if this bit is better than the current best
            if eval_acc >= acc_threshold and eval_asr > best_asr:
                print(f"      Found better bit: {bit_info['layer_name']} idx={bit_info['param_idx']} pos={bit_info['bit_pos']} (ACC: {eval_acc:.4f}, ASR: {eval_asr:.4f})")
                best_asr = eval_asr
                # Store the info needed to perform the flip later
                best_bit_info = {
                    'layer_name': bit_info['layer_name'],
                    'param_idx': bit_info['param_idx'],
                    'bit_pos': bit_info['bit_pos'],
                    'layer_module': bit_info['layer_module'] # Reference to module in ORIGINAL model needed?
                                                                # Let's store the name and find module later.
                }

        if best_bit_info:
            print(f"    B-Step Selected Bit: {best_bit_info['layer_name']} idx={best_bit_info['param_idx']} pos={best_bit_info['bit_pos']} (Best ASR: {best_asr:.4f})")
        else:
            print("    B-Step: No suitable bit found in the sample that meets criteria.")
            
        return best_bit_info
    # --- END MODIFIED _identify_vuln_bit --- 

    def _evaluate_attack(self, model_state, trigger_pattern, dataset, target_class):
        """Evaluate ACC and ASR, applying the trigger for ASR calculation."""
        print("    Evaluating attack state...")
        if len(dataset) == 0:
            print("    Warning: Evaluation dataset is empty.")
            return 0.0, 0.0

        model_state.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        total = 0
        correct_clean = 0
        attack_success_triggered = 0
        
        self.current_trigger_pattern = trigger_pattern # Set trigger for apply_trigger method

        with torch.no_grad():
            for batch in dataloader:
                # --- Get Clean Inputs and Labels --- 
                input_ids, attention_mask, labels, pixel_values = None, None, None, None
                original_batch = batch # Keep for custom_forward_fn if needed

                # Handle different batch structures to find labels and inputs
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask')
                    labels = batch.get('labels')
                    pixel_values = batch.get('pixel_values') # Check for image data
                elif isinstance(batch, (list, tuple)):
                    # Try to handle (inputs, labels) structure
                    if len(batch) >= 2:
                         inputs_data, labels = batch[0], batch[1]
                         if isinstance(inputs_data, dict):
                             input_ids = inputs_data.get('input_ids')
                             attention_mask = inputs_data.get('attention_mask')
                             pixel_values = inputs_data.get('pixel_values')
                         elif isinstance(inputs_data, torch.Tensor):
                              # Assume image if 4D, else NLP input_ids
                              if len(inputs_data.shape) == 4: pixel_values = inputs_data
                              else: input_ids = inputs_data
                    else: continue # Skip if tuple/list format is wrong
                elif isinstance(batch, TensorDataset): # Handle TensorDataset from Dcert
                    # Assume structure (input_ids, attention_mask, labels)
                    if len(batch) == 3:
                         input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
                    else: continue
                else:
                     continue # Skip unsupported formats

                if labels is None: continue # Cannot evaluate without labels
                labels = labels.to(self.device)
                total += labels.size(0)

                # --- Clean Evaluation (ACC) --- 
                try:
                    if self.custom_forward_fn:
                        # Custom forward fn should handle the original batch structure
                        clean_outputs = self.custom_forward_fn(model_state, original_batch)
                    else:
                        # Standard forward pass based on detected input type
                        if input_ids is not None and attention_mask is not None:
                             outputs_logits = model_state(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device)).logits
                        elif pixel_values is not None:
                             outputs_logits = model_state(pixel_values=pixel_values.to(self.device)).logits
                        else: continue # Skip if no valid standard input
                        clean_outputs = outputs_logits # Use logits directly

                    if isinstance(clean_outputs, dict) and 'logits' in clean_outputs:
                        clean_outputs = clean_outputs['logits']
                        
                    _, clean_predicted = clean_outputs.max(1)
                    correct_clean += (clean_predicted == labels).sum().item()
                except Exception as e:
                     print(f"    Error during clean evaluation step: {e}")
                     # Allow loop to continue, but ACC will be lower

                # --- Triggered Evaluation (ASR) --- 
                try:
                    # Apply trigger
                    triggered_batch = self.apply_trigger(original_batch) # Apply trigger to original structure
                    
                    # Get inputs from triggered batch
                    trig_input_ids, trig_attn_mask, trig_pixel_values = None, None, None
                    if isinstance(triggered_batch, dict):
                        trig_input_ids = triggered_batch.get('input_ids')
                        trig_attn_mask = triggered_batch.get('attention_mask')
                        trig_pixel_values = triggered_batch.get('pixel_values')
                    elif isinstance(triggered_batch, (list, tuple)) and len(triggered_batch) >= 2:
                         if isinstance(triggered_batch[0], dict): 
                              trig_input_ids = triggered_batch[0].get('input_ids')
                              trig_attn_mask = triggered_batch[0].get('attention_mask')
                              trig_pixel_values = triggered_batch[0].get('pixel_values')
                         elif isinstance(triggered_batch[0], torch.Tensor):
                              if len(triggered_batch[0].shape) == 4: trig_pixel_values = triggered_batch[0]
                              else: trig_input_ids = triggered_batch[0]
                    elif isinstance(triggered_batch, TensorDataset): # If trigger applied to TensorDataset?
                         if len(triggered_batch) == 3:
                              trig_input_ids, trig_attn_mask, _ = triggered_batch[0], triggered_batch[1], triggered_batch[2]
                    
                    # Run forward pass on triggered data
                    if self.custom_forward_fn:
                        triggered_outputs = self.custom_forward_fn(model_state, triggered_batch)
                    else:
                        if trig_input_ids is not None and trig_attn_mask is not None:
                             outputs_logits = model_state(input_ids=trig_input_ids.to(self.device), attention_mask=trig_attn_mask.to(self.device)).logits
                        elif trig_pixel_values is not None:
                             outputs_logits = model_state(pixel_values=trig_pixel_values.to(self.device)).logits
                        else: continue # Skip if no valid triggered input
                        triggered_outputs = outputs_logits

                    if isinstance(triggered_outputs, dict) and 'logits' in triggered_outputs:
                        triggered_outputs = triggered_outputs['logits']
                        
                    _, triggered_predicted = triggered_outputs.max(1)
                    attack_success_triggered += (triggered_predicted == target_class).sum().item()
                except Exception as e:
                    print(f"    Error during triggered evaluation step: {e}")
                    # Allow loop to continue, but ASR will be lower

        accuracy = correct_clean / total if total > 0 else 0.0
        asr = attack_success_triggered / total if total > 0 else 0.0
        
        print(f"    Evaluation - ACC: {accuracy:.4f}, ASR: {asr:.4f}")
        return accuracy, asr

    # --- Main Bit Identification Method ---

    def identify_bits(self, target_class, acc_threshold, asr_threshold, max_iterations):
        print(f"\nStarting Bit Identification (Target: {target_class}, Max Iter: {max_iterations})...")
        if self.substitute_model is None or self.Dl is None:
            raise RuntimeError("Knowledge Discovery must be run before Bit Identification.")

        optimal_bits_info = [] # List to store info about flipped bits
        current_model_state = copy.deepcopy(self.substitute_model) # Work on a copy
        
        # --- Placeholder for flippable bit constraints ---
        # In a real scenario, load this from memory templating
        flippable_bit_constraints = {} 
        print("    Warning: Using placeholder for flippable bit constraints.")
        # -----------------------------------------------

        # Initialize trigger (Line 2)
        print("Step 1: Initial Trigger Generation...")
        # Pass dataloader for Dl, target_class. Get pattern back.
        if len(self.Dl) == 0:
             print("  Warning: Dataset Dl is empty. Cannot generate initial trigger.")
             current_trigger_pattern = None
        else:
             initial_dataloader = DataLoader(self.Dl, batch_size=32, shuffle=True)
             current_trigger_pattern = self._update_trigger(current_model_state, None, initial_dataloader, target_class)

        # Initial evaluation (Line 3)
        asr, acc = self._evaluate_attack(current_model_state, current_trigger_pattern, self.Dl, target_class)
        print(f"  Initial State - ACC: {acc:.4f}, ASR: {asr:.4f}")

        # Iteration loop (Line 4)
        print("Step 2: Iterative Bit Search (T-step & B-step)...")
        for iter_num in range(max_iterations):
            print(f"\n  Iteration {iter_num + 1}/{max_iterations}")
            if asr >= asr_threshold:
                print("  ASR threshold met. Stopping iterations.")
                break

            # B-step: Identify best bit (Line 5)
            # Pass necessary info including target_class and acc_threshold for internal evaluation
            best_bit_to_add = self._identify_vuln_bit(
                current_model_state, current_trigger_pattern, optimal_bits_info, 
                target_class, acc_threshold
            )
            if best_bit_to_add is None:
                print("  B-Step could not identify a suitable bit. Stopping iterations.")
                break

            # Simulate adding the bit and re-evaluate (Line 6)
            # The evaluation is already done inside the B-step now to select the best bit
            # We just need to apply the *selected* best bit to the current_model_state
            print(f"    Applying selected bit: {best_bit_to_add['layer_name']} idx={best_bit_to_add['param_idx']} pos={best_bit_to_add['bit_pos']}")
            # Need to get the actual module from current_model_state to flip
            try:
                 module_to_flip = current_model_state.get_submodule(best_bit_to_add['layer_name'])
                 layer_info_dict = next((li for li in self.layer_info if li['name'] == best_bit_to_add['layer_name']), None)
                 if layer_info_dict:
                      temp_layer_dict = layer_info_dict.copy()
                      temp_layer_dict['module'] = module_to_flip # Point to module in current_model_state
                      flip_bit(temp_layer_dict, best_bit_to_add['param_idx'], best_bit_to_add['bit_pos'])
                      # Bit is now applied to current_model_state
                      optimal_bits_info.append(best_bit_to_add) # Track the accepted bit
                      print("    Bit successfully applied to current model state.")
                      # Re-evaluate the main state AFTER applying the accepted bit
                      asr, acc = self._evaluate_attack(current_model_state, current_trigger_pattern, self.Dl, target_class)
                      print(f"    State after applying bit - ACC: {acc:.4f}, ASR: {asr:.4f}")
                 else:
                      print(f"    Warning: Could not find layer info for selected bit {best_bit_to_add['layer_name']}. Skipping application.")
            except Exception as e:
                 print(f"    Error applying selected bit {best_bit_to_add}: {e}")
                 # Optionally stop or continue without applying
                 break # Stop if applying the best bit fails
                 
            # T-step: Update trigger (Line 10)
            if len(self.Dl) > 0:
                 current_dataloader = DataLoader(self.Dl, batch_size=32, shuffle=True) # Use Dl for trigger update
                 current_trigger_pattern = self._update_trigger(current_model_state, current_trigger_pattern, current_dataloader, target_class)
            else:
                 print("  Warning: Dataset Dl is empty. Skipping trigger update.")

        print(f"\nBit Identification Complete. Total bits flipped: {len(optimal_bits_info)}")
        return optimal_bits_info, current_trigger_pattern # Return identified bits and final trigger

    def perform_attack(self, target_class, query_budget=3000, acc_threshold=0.8, 
                         asr_threshold=0.9, max_iterations=1000):
        """Orchestrate the Groan attack.
        """
        print("--- Starting Groan Attack --- ")
        start_time = time.time()
        
        # 1. Knowledge Discovery
        self.knowledge_discovery(target_class, query_budget)
        
        # 2. Bit Identification
        identified_bits, final_trigger_pattern = self.identify_bits(
            target_class, acc_threshold, asr_threshold, max_iterations
        )

        # 3. (Simulation only for now) Apply bits to a copy of the original model
        # In a real scenario, this would involve Rowhammer
        print("\nSimulating application of identified bits...")
        attacked_model = copy.deepcopy(self.model) # Start from original
        applied_count = 0
        if identified_bits: # Check if list is not empty
            print(f"  Attempting to flip {len(identified_bits)} identified bits...")
            for bit_info in identified_bits:
                layer_module = bit_info.get('layer_module')
                param_idx = bit_info.get('param_idx')
                bit_pos = bit_info.get('bit_pos')
                
                # Find the corresponding layer_info dict for the module in the *attacked_model*
                # Need to regenerate layer_info for the attacked_model or assume structure is identical
                # For simulation, we assume structure is identical to self.layer_info
                layer_info_dict = next((info for info in self.layer_info if info['name'] == bit_info.get('layer_name')), None)
                
                if layer_info_dict and layer_module is not None and param_idx is not None and bit_pos is not None:
                    # Get the module from the *attacked_model* using the name
                    try:
                        module_to_flip = attacked_model.get_submodule(layer_info_dict['name'])
                        # Create a temporary layer dict pointing to the module in attacked_model
                        temp_layer_dict = layer_info_dict.copy()
                        temp_layer_dict['module'] = module_to_flip
                        
                        flip_bit(temp_layer_dict, param_idx, bit_pos) # Flip bit in attacked_model
                        applied_count += 1
                    except Exception as e:
                        print(f"    Error applying flip for bit {bit_info}: {e}")
                else:
                    print(f"    Skipping invalid bit info: {bit_info}")
        else:
            print("  No bits identified to flip.")
            
        print(f"  Applied {applied_count} simulated bit flips to final model.")

        # 4. Final Evaluation (Placeholder)
        print("\nEvaluating final attacked model (Placeholder Evaluation)...")
        # Use the main dataset for final evaluation? Or Dl?
        final_acc, final_asr = self._evaluate_attack(attacked_model, final_trigger_pattern, self.dataset, target_class)

        end_time = time.time()
        results = {
            'attack_type': 'Groan',
            'target_class': target_class,
            'identified_bits_count': len(identified_bits),
            'applied_flips_count': applied_count,
            'final_accuracy': final_acc, # Placeholder value
            'final_asr': final_asr,       # Placeholder value
            'final_trigger_pattern_shape': final_trigger_pattern.shape if final_trigger_pattern is not None else None,
            'execution_time': end_time - start_time
        }
        print("--- Groan Attack Complete --- ")
        return results

# Example Usage (Placeholder)
if __name__ == '__main__':
    print("Running Groan Attack Script Example")

    # Basic Configuration
    MODEL_NAME = "prajjwal1/bert-tiny" # Use a small model for testing
    TARGET_CLASS = 1 # Example target class
    QUERY_BUDGET = 1000 # Reduced query budget for faster testing
    MAX_ITERATIONS = 50 # Reduced iterations for faster testing
    ACC_THRESHOLD = 0.7 # Example accuracy threshold
    ASR_THRESHOLD = 0.6 # Example ASR threshold

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model and Tokenizer
    try:
        print(f"Loading model: {MODEL_NAME}")
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
        model.eval() # Ensure model is in eval mode initially
        print(f"Loading tokenizer: {MODEL_NAME}")
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}. Exiting.")
        exit()

    # Load/Create Dataset
    # Use the importer helper from umup_attack_example (requires correct path setup)
    if load_or_create_dataset is None or PIIDataset is None:
        print("Error: Cannot load dataset utilities. Exiting.")
        exit()
        
    try:
        print("Loading/Creating dataset...")
        # Use smaller dataset for quick testing
        dataset = load_or_create_dataset(tokenizer, num_pii=500, num_non_pii=500) 
        if len(dataset) == 0:
             raise ValueError("Loaded dataset is empty.")
    except Exception as e:
        print(f"Error loading/creating dataset: {e}. Exiting.")
        exit()

    # Instantiate GroanAttack
    # Assuming the custom_forward_fn from umup_attack_example is suitable or None
    # Let's use None for simplicity first
    print("Instantiating GroanAttack...")
    groan_attack = GroanAttack(model, dataset, device=device, custom_forward_fn=None)

    # Perform the attack
    try:
        print("Performing Groan attack...")
        results = groan_attack.perform_attack(
            target_class=TARGET_CLASS,
            query_budget=QUERY_BUDGET,
            acc_threshold=ACC_THRESHOLD,
            asr_threshold=ASR_THRESHOLD,
            max_iterations=MAX_ITERATIONS
        )
        
        print("\n--- Final Groan Attack Results ---")
        import json
        print(json.dumps(results, indent=2, default=str))

    except Exception as e:
        print(f"\n--- ERROR during Groan attack execution ---")
        traceback.print_exc()
    pass 