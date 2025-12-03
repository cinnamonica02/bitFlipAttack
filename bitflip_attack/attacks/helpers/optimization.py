"""
Optimization module for bit flip attacks

This module contains genetic algorithm implementation for optimizing bit flips.
"""
import numpy as np
import torch
import logging
from bitflip_attack.attacks.helpers.evaluation import evaluate_individual_fitness
from bitflip_attack.attacks.helpers.bit_manipulation import flip_bit
import time

# Set up logger
logger = logging.getLogger(__name__)


def genetic_optimization(model, dataset, candidates, layer_info, target_class, 
                        attack_mode, max_bit_flips, pop_size=50, generations=20, 
                        accuracy_threshold=0.5, device='cuda', custom_forward_fn=None):
    """
    Apply genetic optimization to find optimal bit flips.
    
    Args:
        model: The model being attacked
        dataset: Dataset for evaluation
        candidates: List of bit flip candidates
        layer_info: List of layer information dictionaries
        target_class: Target class for attack
        attack_mode: 'targeted' or 'untargeted'
        max_bit_flips: Maximum number of bits to flip
        pop_size: Population size for genetic algorithm
        generations: Number of generations
        accuracy_threshold: Minimum accuracy to maintain
        device: Device to run evaluation on
        custom_forward_fn: Custom forward function for the model
        
    Returns:
        best_solution: Best set of bits to flip
        flip_history: History of bit flips
    """
    # Store original model weights
    original_weights = {}
    for layer in layer_info:
        original_weights[layer['name']] = layer['module'].weight.data.clone()
    
    # Initialize population (each individual is a subset of candidate bits)
    # Better initialization: explore diverse bit counts from 3 to max_bit_flips
    population = []
    min_bits = 3  # Start with at least 3 bits for meaningful attack
    for i in range(pop_size):
        # Vary the number of bits across the population for diversity
        # First quarter: smaller solutions (3-5 bits)
        # Second quarter: medium solutions (5-10 bits)
        # Third quarter: larger solutions (10-max bits)
        # Fourth quarter: random mix
        if i < pop_size // 4:
            n_bits = np.random.randint(min_bits, min(6, max_bit_flips) + 1)
        elif i < pop_size // 2:
            n_bits = np.random.randint(5, min(11, max_bit_flips) + 1)
        elif i < 3 * pop_size // 4:
            n_bits = np.random.randint(max(8, max_bit_flips // 2), max_bit_flips + 1)
        else:
            n_bits = np.random.randint(min_bits, max_bit_flips + 1)

        n_bits = min(n_bits, len(candidates))  # Don't exceed candidates
        individual = np.random.choice(len(candidates), size=n_bits, replace=False)
        population.append(sorted(individual.tolist()))
    
    # Track best solution
    best_fitness = -float('inf')
    best_solution = None
    best_asr = 0
    best_accuracy = 0

    # Convergence tracking
    generations_without_improvement = 0
    max_stagnant_generations = 4  # Stop if no improvement for 4 generations
    previous_best_asr = 0

    # Store bit flip history for final report
    flip_history = []

    for gen in range(generations):
        logger.info(f"Starting Generation {gen+1}/{generations}")
        fitness_scores = []
        accuracies = []
        asrs = []
        
        # Evaluate each individual
        logger.info(f"Evaluating {len(population)} individuals...")
        for i, individual in enumerate(population):
            logger.debug(f"  Evaluating individual {i+1}/{len(population)} (Size: {len(individual)} bits)")
            start_eval_time = time.time()
            
            # Restore original weights
            # print("    Restoring original weights...") # DEBUG (Optional, can be verbose)
            for layer in layer_info:
                layer['module'].weight.data.copy_(original_weights[layer['name']])
            
            # Evaluate individual
            # print("    Calling evaluate_individual_fitness...") # DEBUG (Optional)
            fitness, accuracy, asr = evaluate_individual_fitness(
                model, dataset, individual, candidates, layer_info,
                target_class, attack_mode, accuracy_threshold,
                device, custom_forward_fn
            )
            eval_time = time.time() - start_eval_time
            logger.debug(f"    Individual {i+1} - Fitness: {fitness:.4f}, Acc: {accuracy:.4f}, "
                        f"ASR: {asr:.4f}, Time: {eval_time:.2f}s")
            
            fitness_scores.append(fitness)
            accuracies.append(accuracy)
            asrs.append(asr)
            
            # Check if this is the best solution so far
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual.copy()
                best_asr = asr
                best_accuracy = accuracy
                
                # Store bit flip information for the best solution
                flip_history = []
                for idx in individual:
                    candidate = candidates[idx]
                    layer_idx = candidate['layer_idx']
                    layer = layer_info[layer_idx] if layer_idx >= 0 else find_layer_by_name(layer_info, candidate['layer_name'])
                    param_idx = candidate['parameter_idx']
                    bit_pos = candidate['bit_position']
                    coords = candidate['coords']
                    
                    flip_history.append({
                        'layer': layer['name'],
                        'parameter_idx': param_idx,
                        'coords': coords,
                        'bit_position': bit_pos,
                    })
        
        # Log progress
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        avg_asr = sum(asrs) / len(asrs)
        avg_acc = sum(accuracies) / len(accuracies)
        logger.info(f"Generation {gen+1}/{generations}: Avg ASR = {avg_asr:.4f}, "
                   f"Avg Acc = {avg_acc:.4f}, Best ASR = {best_asr:.4f}")

        # Check for convergence (no improvement)
        improvement_threshold = 0.01  # 1% improvement considered significant
        if best_asr > previous_best_asr + improvement_threshold:
            generations_without_improvement = 0
            previous_best_asr = best_asr
            logger.info(f"  -> New best ASR: {best_asr:.4f}")
        else:
            generations_without_improvement += 1
            logger.info(f"  -> No significant improvement ({generations_without_improvement}/{max_stagnant_generations})")

        # Early stopping conditions
        # 1. Achieved very high ASR (above target)
        if best_asr > 0.75:
            logger.info(f"Early stopping at generation {gen+1}: High ASR achieved ({best_asr:.4f})")
            break

        # 2. Convergence: no improvement for several generations
        if generations_without_improvement >= max_stagnant_generations:
            logger.info(f"Early stopping at generation {gen+1}: Convergence detected "
                       f"(no improvement for {max_stagnant_generations} generations)")
            break
        
        # Generate next population
        logger.debug(f"Creating next generation ({gen+1})")
        start_next_gen_time = time.time()
        next_population = create_next_generation(
            population, fitness_scores, pop_size,
            candidates, max_bit_flips
        )
        next_gen_time = time.time() - start_next_gen_time
        logger.debug(f"Next generation created in {next_gen_time:.2f}s")
        
        population = next_population
    
    # Restore original weights (for final application of best solution)
    for layer in layer_info:
        layer['module'].weight.data.copy_(original_weights[layer['name']])
    
    return best_solution, flip_history


def create_next_generation(population, fitness_scores, pop_size, candidates, max_bit_flips):
    """
    Create next generation using selection, crossover, and mutation.
    
    Args:
        population: Current population
        fitness_scores: Fitness scores for current population
        pop_size: Population size
        candidates: List of bit flip candidates
        max_bit_flips: Maximum number of bits to flip
        
    Returns:
        next_population: Next generation population
    """
    # Get index of best individual
    best_idx = np.argmax(fitness_scores)
    best_individual = population[best_idx]
    
    # Create next population with elitism (keep best solution)
    next_population = [best_individual.copy()]
    
    # Tournament selection and crossover to fill rest of population
    while len(next_population) < pop_size:
        # Tournament selection
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)
        
        # Crossover
        child = crossover(parent1, parent2)
        
        # Mutation
        child = mutation(child, candidates, max_bit_flips)
        
        # Add to next population
        next_population.append(child)
    
    return next_population


def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Select an individual using tournament selection.
    
    Args:
        population: List of individuals
        fitness_scores: List of fitness scores
        tournament_size: Size of tournament
        
    Returns:
        selected: Selected individual
    """
    indices = np.random.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    tournament_fitness = [fitness_scores[i] for i in indices]
    winner_idx = indices[np.argmax(tournament_fitness)]
    return population[winner_idx]


def crossover(parent1, parent2):
    """
    Perform crossover between two parents.
    
    Args:
        parent1: First parent (list of indices)
        parent2: Second parent (list of indices)
        
    Returns:
        child: Resulting child
    """
    if not parent1 or not parent2:
        # Handle empty parents
        return parent1.copy() if parent1 else parent2.copy()
    
    # Single-point crossover
    crossover_point = np.random.randint(0, min(len(parent1), len(parent2)) + 1)
    
    # Create child by combining parts of both parents
    child = sorted(list(set(parent1[:crossover_point] + parent2[crossover_point:])))
    
    return child


def mutation(individual, candidates, max_bit_flips, mutation_rate=0.2):
    """
    Apply mutation to an individual.
    
    Args:
        individual: Individual to mutate
        candidates: List of all candidates
        max_bit_flips: Maximum number of bits to flip
        mutation_rate: Probability of mutation
        
    Returns:
        mutated: Mutated individual
    """
    # Deep copy the individual
    mutated = individual.copy()
    
    # Apply mutation with probability mutation_rate
    if np.random.random() < mutation_rate:
        if len(mutated) > 1 and np.random.random() < 0.5 and len(mutated) >= max_bit_flips // 2:
            # Remove a random bit
            idx_to_remove = np.random.randint(0, len(mutated))
            mutated.pop(idx_to_remove)
        else:
            # Add a random bit if not at max size
            if len(mutated) < max_bit_flips:
                available_bits = [i for i in range(len(candidates)) if i not in mutated]
                if available_bits:
                    bit_to_add = np.random.choice(available_bits)
                    mutated.append(bit_to_add)
                    mutated.sort()
    
    return mutated


def find_layer_by_name(layer_info, name):
    """Find layer in layer_info by name"""
    for layer in layer_info:
        if layer['name'] == name:
            return layer
    return None 