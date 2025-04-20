import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

class BitFlipAttack:
    """
    Implementation of the Bit Flipping Attack (BFA) on deep neural networks.
    This attack identifies and flips critical bits in model parameters to induce
    misclassification with minimal changes to the model.
    """
    
    def __init__(self, model, dataset, target_asr=0.9, max_bit_flips=100, 
                 accuracy_threshold=0.05, device='cuda', 
                 attack_mode='targeted', layer_sensitivity=True, 
                 hybrid_sensitivity=True, alpha=0.5):
        """
        Initialize the bit flipping attack.
        
        Args:
            model: The target model to attack
            dataset: Dataset for evaluating model performance
            target_asr: Target attack success rate
            max_bit_flips: Maximum number of bits to flip
            accuracy_threshold: Maximum acceptable accuracy drop
            device: Device to run the attack on (cuda or cpu)
            attack_mode: 'targeted' or 'untargeted'
            layer_sensitivity: Whether to perform layer sensitivity analysis
            hybrid_sensitivity: Whether to use hybrid sensitivity metric
            alpha: Weight for hybrid sensitivity metric (0 for mag-only, 1 for grad-only)
        """
        self.model = model
        self.dataset = dataset
        self.target_asr = target_asr
        self.max_bit_flips = max_bit_flips
        self.accuracy_threshold = accuracy_threshold
        self.device = device
        self.attack_mode = attack_mode
        self.layer_sensitivity = layer_sensitivity
        self.hybrid_sensitivity = hybrid_sensitivity
        self.alpha = alpha
        
        # Place model on the specified device
        self.model.to(self.device)
        
        # Track attack metrics
        self.original_accuracy = 0.0
        self.final_accuracy = 0.0
        self.initial_asr = 0.0
        self.final_asr = 0.0
        self.bits_flipped = 0
        self.flipped_bits_info = []
        
        # Store the model's layer structure for sensitivity analysis
        self.layer_info = self._get_layer_info()
    
    def _get_layer_info(self):
        """
        Analyze model structure and identify quantized or attackable layers.
        
        Returns:
            List of dictionaries containing layer information
        """
        layer_info = []
        for name, module in self.model.named_modules():
            # Skip containers and non-parameter layers
            if len(list(module.children())) > 0:
                continue
                
            # Focus on layers with weights
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
                    'is_quantized': hasattr(module, 'weight_fake_quant') 
                                  or 'quantized' in module.__class__.__name__.lower()
                })
        
        return layer_info
    
    def _compute_sensitivity(self, layer, inputs, targets, use_gradient=True, use_magnitude=True):
        """
        Compute sensitivity of parameters in a layer using the hybrid metric.
        
        Args:
            layer: Layer to compute sensitivity for
            inputs: Batch of inputs for gradient computation
            targets: Corresponding labels
            use_gradient: Whether to use gradient information
            use_magnitude: Whether to use weight magnitude
            
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
        weight.grad = None  # Clear gradients
        
        # Forward pass
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Check if gradient exists
        if weight.grad is None:
            return weight_magnitude
        
        gradient_magnitude = torch.abs(weight.grad.detach())
        
        # Normalize weights and gradients
        weight_norm = (weight_magnitude - weight_magnitude.min()) / (weight_magnitude.max() - weight_magnitude.min() + 1e-8)
        grad_norm = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)
        
        # Compute hybrid sensitivity
        if use_magnitude and use_gradient:
            sensitivity = self.alpha * grad_norm + (1 - self.alpha) * weight_norm
        elif use_gradient:
            sensitivity = grad_norm
        else:
            sensitivity = weight_norm
            
        return sensitivity
    
    def _rank_layers_by_sensitivity(self, n_samples=100):
        """
        Rank layers by their sensitivity to bit flips.
        
        Args:
            n_samples: Number of samples to use for gradient computation
            
        Returns:
            sorted_layers: List of layer info sorted by sensitivity
        """
        # Get a batch of data for gradient computation
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=n_samples, shuffle=True
        )
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Put model in evaluation mode but enable grad tracking
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True
            
        layer_losses = []
        
        # For each layer, compute sensitivity and induce perturbations
        for idx, layer in enumerate(self.layer_info):
            # Compute sensitivity score
            sensitivity = self._compute_sensitivity(
                layer, inputs, targets, 
                use_gradient=self.hybrid_sensitivity, 
                use_magnitude=True
            )
            
            # Sample bits to flip based on sensitivity
            k = int(layer['n_params'] * 0.001)  # Flip 0.1% of bits for test
            k = max(1, min(k, 100))  # Cap between 1 and 100 bits
            
            # Get top-k indices by sensitivity
            _, indices = torch.topk(sensitivity.view(-1), k)
            
            # Store original weights
            orig_weight = layer['module'].weight.data.clone()
            
            # Apply bit flips to MSB (most significant bit) of selected weights
            for idx in indices:
                # Convert flat index to tensor coordinates
                coords = np.unravel_index(idx.item(), layer['shape'])
                coords = tuple(coords)
                
                # Bit flip at MSB (we'll flip the sign bit for most impact)
                value = layer['module'].weight.data[coords].item()
                # Simple flip for testing: flip the sign to maximize impact
                layer['module'].weight.data[coords] = -value
            
            # Compute loss after perturbation
            outputs = self.model(inputs)
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
            
        # Sort layers by loss (higher loss = more sensitive)
        sorted_layers = sorted(layer_losses, key=lambda x: x['loss'], reverse=True)
        
        # Reset model gradients and mode
        self.model.zero_grad()
        
        return sorted_layers
    
    def _select_bit_candidates(self, layer, n_candidates=1000):
        """
        Select candidate bits for flipping based on sensitivity.
        
        Args:
            layer: Layer information dictionary
            n_candidates: Number of bit candidates to select
            
        Returns:
            candidates: List of candidate bits (indices, positions)
        """
        # Get a batch of data for gradient computation
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=True
        )
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Compute sensitivity
        sensitivity = self._compute_sensitivity(
            layer, inputs, targets, 
            use_gradient=self.hybrid_sensitivity, 
            use_magnitude=True
        )
        
        # Get top-k indices by sensitivity
        n_candidates = min(n_candidates, layer['n_params'])
        _, indices = torch.topk(sensitivity.view(-1), n_candidates)
        
        # For each parameter, consider different bit positions (focus on MSBs)
        candidates = []
        
        # Default to float32 (typical case)
        bit_width = 32
        
        # Determine actual bit width of parameters
        if hasattr(layer['module'], 'weight_fake_quant'):
            # For quantized models with fake quantization
            bit_width = layer['module'].weight_fake_quant.bit_width
        elif layer['is_quantized']:
            # For models with explicit quantization
            bit_width = 8  # Assume int8 for quantized layers
        
        # For each selected parameter, add MSBs as candidates
        for idx in indices:
            # Convert flat index to tensor coordinates
            flat_idx = idx.item()
            coords = np.unravel_index(flat_idx, layer['shape'])
            
            # Add MSB and sign bit as candidates
            if bit_width == 32:  # FP32
                # For floating point, prioritize sign bit and exponent bits
                bit_positions = [31]  # Sign bit
                bit_positions.extend(range(30, 22, -1))  # Exponent bits
            elif bit_width == 16:  # FP16
                bit_positions = [15]  # Sign bit
                bit_positions.extend(range(14, 9, -1))  # Exponent bits
            else:  # Integer/quantized
                # Prioritize MSBs
                bit_positions = list(range(bit_width-1, bit_width-4, -1))
            
            for bit_pos in bit_positions:
                candidates.append({
                    'layer_name': layer['name'],
                    'layer_idx': self.layer_info.index(layer),
                    'parameter_idx': flat_idx,
                    'coords': coords,
                    'bit_position': bit_pos,
                    'sensitivity': sensitivity.view(-1)[flat_idx].item()
                })
        
        return candidates
    
    def _flip_bit(self, layer, param_idx, bit_pos):
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
        
        # Convert to binary representation
        if isinstance(value, float):
            # For floating point values, use bit-level manipulation
            # Convert float to its bit representation
            int_repr = torch.tensor([value], dtype=torch.float32).view(torch.int32).item()
            # Flip the specific bit
            int_repr ^= (1 << bit_pos)
            # Convert back to float
            new_value = torch.tensor([int_repr], dtype=torch.int32).view(torch.float32).item()
        else:
            # For integer values (quantized models)
            # Flip the bit directly
            new_value = value ^ (1 << bit_pos)
        
        # Update the parameter
        weight.data[coords] = new_value
        
        return old_value, new_value
    
    def _evaluate(self, target_class=None):
        """
        Evaluate model performance and attack success rate.
        
        Args:
            target_class: Target class for targeted attacks (None for untargeted)
            
        Returns:
            accuracy: Model accuracy on clean test set
            asr: Attack success rate
        """
        self.model.eval()
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=False
        )
        
        correct = 0
        attack_success = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                # Calculate accuracy
                correct += (predicted == targets).sum().item()
                
                # Calculate attack success rate
                if self.attack_mode == 'targeted' and target_class is not None:
                    # For targeted attacks: success if model predicts target class
                    attack_success += (predicted == target_class).sum().item()
                else:
                    # For untargeted attacks: success if model prediction is wrong
                    attack_success += (predicted != targets).sum().item()
                
                total += targets.size(0)
        
        accuracy = correct / total
        asr = attack_success / total
        
        return accuracy, asr
    
    def _genetic_optimization(self, candidates, target_class, pop_size=50, generations=20, 
                             accuracy_threshold=None):
        """
        Apply genetic optimization to find optimal bit flips.
        
        Args:
            candidates: List of bit flip candidates
            target_class: Target class for attack
            pop_size: Population size for genetic algorithm
            generations: Number of generations
            accuracy_threshold: Minimum accuracy to maintain
            
        Returns:
            best_solution: Best set of bits to flip
        """
        if accuracy_threshold is None:
            accuracy_threshold = self.accuracy_threshold
            
        # Initialize population (each individual is a subset of candidate bits)
        population = []
        for _ in range(pop_size):
            # Randomly select bits to include (max 10% of candidates)
            n_bits = min(self.max_bit_flips, np.random.randint(1, max(2, len(candidates) // 10)))
            individual = np.random.choice(len(candidates), size=n_bits, replace=False)
            population.append(sorted(individual.tolist()))
        
        # Track best solution
        best_fitness = -float('inf')
        best_solution = None
        best_asr = 0
        best_accuracy = 0
        
        # Store original model weights
        original_weights = {}
        for layer in self.layer_info:
            original_weights[layer['name']] = layer['module'].weight.data.clone()
        
        # Store bit flip history for final report
        flip_history = []
        
        for gen in range(generations):
            fitness_scores = []
            
            # Evaluate each individual
            for individual in population:
                # Restore original weights
                for layer in self.layer_info:
                    layer['module'].weight.data.copy_(original_weights[layer['name']])
                
                # Apply bit flips for this individual
                for idx in individual:
                    candidate = candidates[idx]
                    layer_idx = candidate['layer_idx']
                    layer = self.layer_info[layer_idx]
                    param_idx = candidate['parameter_idx']
                    bit_pos = candidate['bit_position']
                    
                    old_val, new_val = self._flip_bit(layer, param_idx, bit_pos)
                    
                    # Store bit flip for history
                    flip_info = {
                        'layer': layer['name'],
                        'parameter_idx': param_idx,
                        'bit_position': bit_pos,
                        'old_value': old_val,
                        'new_value': new_val
                    }
                    if idx not in flip_history:
                        flip_history.append(flip_info)
                
                # Evaluate model
                accuracy, asr = self._evaluate(target_class)
                
                # Compute fitness: maximize ASR while keeping accuracy above threshold
                if accuracy >= accuracy_threshold:
                    # If accuracy is acceptable, focus on maximizing ASR
                    fitness = asr
                else:
                    # If accuracy is below threshold, penalize based on accuracy drop
                    fitness = asr - (accuracy_threshold - accuracy)
                
                fitness_scores.append(fitness)
                
                # Check if this is the best solution so far
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
                    best_asr = asr
                    best_accuracy = accuracy
            
            # Early stopping if we've achieved target ASR
            if best_asr >= self.target_asr:
                break
                
            # Select parents for next generation (tournament selection)
            next_population = [best_solution]  # Elitism: keep best solution
            
            while len(next_population) < pop_size:
                # Tournament selection
                idx1, idx2 = np.random.choice(pop_size, size=2, replace=False)
                parent1 = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
                
                idx1, idx2 = np.random.choice(pop_size, size=2, replace=False)
                parent2 = population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]
                
                # Crossover
                crossover_point = np.random.randint(1, min(len(parent1), len(parent2)))
                child = sorted(list(set(parent1[:crossover_point] + parent2[crossover_point:])))
                
                # Mutation: add or remove a random bit
                if np.random.random() < 0.2:  # 20% mutation rate
                    if len(child) > 1 and np.random.random() < 0.5:
                        # Remove a random bit
                        idx_to_remove = np.random.choice(len(child))
                        child.pop(idx_to_remove)
                    else:
                        # Add a random bit
                        available_bits = [i for i in range(len(candidates)) if i not in child]
                        if available_bits:
                            bit_to_add = np.random.choice(available_bits)
                            child.append(bit_to_add)
                            child.sort()
                
                next_population.append(child)
            
            # Update population
            population = next_population
            
        # Return best solution
        return best_solution, flip_history
    
    def perform_attack(self, target_class=0, num_candidates=1000):
        """
        Perform the bit flipping attack.
        
        Args:
            target_class: Target class for targeted attacks
            num_candidates: Number of bit candidates to evaluate
            
        Returns:
            results: Dictionary with attack results
        """
        print(f"Starting bit flipping attack...")
        start_time = time.time()
        
        # Evaluate initial model performance
        self.original_accuracy, self.initial_asr = self._evaluate(target_class)
        print(f"Initial model accuracy: {self.original_accuracy:.4f}")
        print(f"Initial attack success rate: {self.initial_asr:.4f}")
        
        # Perform layer sensitivity analysis if enabled
        if self.layer_sensitivity:
            print("Performing layer sensitivity analysis...")
            sensitive_layers = self._rank_layers_by_sensitivity()
            
            # Print top 5 most sensitive layers
            print("\nTop 5 most sensitive layers:")
            for i, layer in enumerate(sensitive_layers[:5]):
                print(f"{i+1}. {layer['name']} ({layer['type']}): Loss = {layer['loss']:.4f}")
            
            # Focus on the most sensitive layer
            target_layer = self.layer_info[sensitive_layers[0]['index']]
        else:
            # If not using sensitivity analysis, choose a layer with large parameter count
            sorted_layers = sorted(self.layer_info, key=lambda x: x['n_params'], reverse=True)
            target_layer = sorted_layers[0]
        
        print(f"\nTargeting layer: {target_layer['name']} ({target_layer['type']})")
        print(f"Number of parameters: {target_layer['n_params']}")
        
        # Select candidate bits for flipping
        print(f"Selecting {num_candidates} bit candidates...")
        candidates = self._select_bit_candidates(target_layer, n_candidates=num_candidates)
        
        # Optimize bit selection using genetic algorithm
        print("Optimizing bit selection using genetic algorithm...")
        best_solution, flip_history = self._genetic_optimization(
            candidates, target_class, 
            pop_size=50, generations=20,
            accuracy_threshold=self.original_accuracy - self.accuracy_threshold
        )
        
        # Apply the best solution to the model
        print("Applying optimal bit flips...")
        
        # Restore original model
        self.model.load_state_dict(torch.load('temp_original_model.pt'))
        
        # Apply bit flips
        flipped_bits = []
        for idx in best_solution:
            candidate = candidates[idx]
            layer_idx = candidate['layer_idx']
            layer = self.layer_info[layer_idx]
            param_idx = candidate['parameter_idx']
            bit_pos = candidate['bit_position']
            
            old_val, new_val = self._flip_bit(layer, param_idx, bit_pos)
            
            # Convert flat index to tensor coordinates
            coords = np.unravel_index(param_idx, layer['module'].weight.shape)
            
            # Store bit flip information
            flipped_bits.append({
                'Layer': layer['name'],
                'Parameter': f"{coords}",
                'Bit Position': bit_pos,
                'Original Value': old_val,
                'New Value': new_val
            })
        
        # Evaluate final model performance
        self.final_accuracy, self.final_asr = self._evaluate(target_class)
        print(f"Final model accuracy: {self.final_accuracy:.4f}")
        print(f"Final attack success rate: {self.final_asr:.4f}")
        
        # Track number of bits flipped
        self.bits_flipped = len(best_solution)
        self.flipped_bits_info = flipped_bits
        
        # Compute attack metrics
        accuracy_drop = self.original_accuracy - self.final_accuracy
        asr_improvement = self.final_asr - self.initial_asr
        
        print(f"Accuracy drop: {accuracy_drop:.4f}")
        print(f"ASR improvement: {asr_improvement:.4f}")
        print(f"Number of bits flipped: {self.bits_flipped}")
        
        # Compile results
        results = {
            'original_accuracy': self.original_accuracy,
            'final_accuracy': self.final_accuracy,
            'accuracy_drop': accuracy_drop,
            'initial_asr': self.initial_asr,
            'final_asr': self.final_asr,
            'asr_improvement': asr_improvement,
            'bits_flipped': self.bits_flipped,
            'flipped_bits': flipped_bits,
            'target_class': target_class,
            'execution_time': time.time() - start_time
        }
        
        return results
    
    def save_results(self, results, output_dir="results"):
        """
        Save attack results to files.
        
        Args:
            results: Dictionary with attack results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save attack metrics
        metrics = [
            {'Metric': 'Original Accuracy', 'Value': results['original_accuracy']},
            {'Metric': 'Final Accuracy', 'Value': results['final_accuracy']},
            {'Metric': 'Accuracy Drop', 'Value': results['accuracy_drop']},
            {'Metric': 'Initial ASR', 'Value': results['initial_asr']},
            {'Metric': 'Final ASR', 'Value': results['final_asr']},
            {'Metric': 'ASR Improvement', 'Value': results['asr_improvement']},
            {'Metric': 'Bits Flipped', 'Value': results['bits_flipped']},
            {'Metric': 'Execution Time (s)', 'Value': results['execution_time']}
        ]
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, f"attack_results_{timestamp}.csv"), index=False)
        
        # Save flipped bits details
        flipped_bits_df = pd.DataFrame(results['flipped_bits'])
        flipped_bits_df.to_csv(os.path.join(output_dir, f"flipped_bits_{timestamp}.csv"), index=False)
        
        # Generate ASR vs Accuracy plot
        self._plot_asr_accuracy(
            results['initial_asr'], results['final_asr'],
            results['original_accuracy'], results['final_accuracy'],
            timestamp, output_dir
        )
        
        print(f"Results saved to {output_dir}/")
    
    def _plot_asr_accuracy(self, initial_asr, final_asr, original_acc, final_acc, timestamp, output_dir):
        """
        Generate a plot showing ASR and accuracy before and after attack.
        
        Args:
            initial_asr: Initial attack success rate
            final_asr: Final attack success rate
            original_acc: Original model accuracy
            final_acc: Final model accuracy
            timestamp: Timestamp for filename
            output_dir: Directory to save plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot ASR
        plt.plot(['Initial', 'Final'], [initial_asr, final_asr], 'ro-', label='Attack Success Rate (ASR)')
        
        # Plot Accuracy
        plt.plot(['Initial', 'Final'], [original_acc, final_acc], 'bo-', label='Model Accuracy')
        
        plt.title('Attack Success Rate and Model Accuracy')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"asr_accuracy_plot_{timestamp}.png"), dpi=300)
        plt.close()
