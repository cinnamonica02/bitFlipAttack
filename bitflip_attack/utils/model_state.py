"""
Model State Management Utilities

This module provides utilities for saving, loading, and comparing model states
during bit flip attacks to ensure proper evaluation and avoid contamination
between baseline and attacked model assessments.
"""

import torch
import copy
import os
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelStateManager:
    """
    Manages model state preservation and restoration for attack evaluation.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Initialize the ModelStateManager.

        Args:
            model: PyTorch model to manage
        """
        self.model = model
        self.saved_states = {}

    def save_state(self, state_name: str = "baseline") -> Dict[str, torch.Tensor]:
        """
        Save the current model state with a given name.

        Args:
            state_name: Identifier for this state (e.g., 'baseline', 'attacked')

        Returns:
            Dictionary containing the saved state
        """
        logger.info(f"Saving model state: {state_name}")

        # Create a deep copy of the state dict
        state_dict = {}
        for name, param in self.model.state_dict().items():
            if param is not None:
                # Clone the parameter to create independent copy
                state_dict[name] = param.clone().detach()
            else:
                state_dict[name] = None

        self.saved_states[state_name] = state_dict

        # Log state information
        total_params = sum(p.numel() for p in state_dict.values() if p is not None)
        logger.info(f"Saved state '{state_name}' with {len(state_dict)} parameters ({total_params:,} elements)")

        return state_dict

    def restore_state(self, state_name: str = "baseline") -> None:
        """
        Restore a previously saved model state.

        Args:
            state_name: Identifier of the state to restore

        Raises:
            ValueError: If the specified state doesn't exist
        """
        if state_name not in self.saved_states:
            raise ValueError(f"State '{state_name}' not found. Available states: {list(self.saved_states.keys())}")

        logger.info(f"Restoring model state: {state_name}")

        # Load the saved state
        saved_state = self.saved_states[state_name]

        # Restore parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in saved_state and saved_state[name] is not None:
                    param.copy_(saved_state[name])

        # Restore buffers (e.g., batch norm running stats)
        for name, buffer in self.model.named_buffers():
            if name in saved_state and saved_state[name] is not None:
                buffer.copy_(saved_state[name])

        logger.info(f"Successfully restored state '{state_name}'")

    def compare_states(self, state_name1: str, state_name2: str) -> Dict[str, Any]:
        """
        Compare two saved model states and report differences.

        Args:
            state_name1: First state to compare
            state_name2: Second state to compare

        Returns:
            Dictionary with comparison statistics
        """
        if state_name1 not in self.saved_states:
            raise ValueError(f"State '{state_name1}' not found")
        if state_name2 not in self.saved_states:
            raise ValueError(f"State '{state_name2}' not found")

        state1 = self.saved_states[state_name1]
        state2 = self.saved_states[state_name2]

        differences = {}
        total_params = 0
        changed_params = 0
        total_diff = 0.0
        max_diff = 0.0
        changed_layers = []

        for name in state1.keys():
            if name not in state2:
                continue

            param1 = state1[name]
            param2 = state2[name]

            if param1 is None or param2 is None:
                continue

            # Calculate differences
            diff = torch.abs(param1 - param2)
            num_changed = (diff > 0).sum().item()

            if num_changed > 0:
                changed_layers.append(name)
                changed_params += num_changed
                total_diff += diff.sum().item()
                max_diff = max(max_diff, diff.max().item())

                differences[name] = {
                    'num_changed': num_changed,
                    'total_elements': param1.numel(),
                    'percent_changed': 100.0 * num_changed / param1.numel(),
                    'mean_diff': diff.sum().item() / num_changed if num_changed > 0 else 0.0,
                    'max_diff': diff.max().item()
                }

            total_params += param1.numel()

        comparison_stats = {
            'total_parameters': total_params,
            'changed_parameters': changed_params,
            'percent_changed': 100.0 * changed_params / total_params if total_params > 0 else 0.0,
            'total_diff': total_diff,
            'max_diff': max_diff,
            'num_changed_layers': len(changed_layers),
            'changed_layers': changed_layers,
            'layer_details': differences
        }

        logger.info(f"Comparison {state_name1} vs {state_name2}:")
        logger.info(f"  - Changed parameters: {changed_params:,} / {total_params:,} ({comparison_stats['percent_changed']:.4f}%)")
        logger.info(f"  - Changed layers: {len(changed_layers)}")
        logger.info(f"  - Max difference: {max_diff:.6e}")

        return comparison_stats

    def save_to_disk(self, filepath: str, state_name: str = "baseline") -> None:
        """
        Save a model state to disk.

        Args:
            filepath: Path to save the state
            state_name: Name of the state to save
        """
        if state_name not in self.saved_states:
            raise ValueError(f"State '{state_name}' not found")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.saved_states[state_name], filepath)
        logger.info(f"Saved state '{state_name}' to {filepath}")

    def load_from_disk(self, filepath: str, state_name: str = "loaded") -> None:
        """
        Load a model state from disk.

        Args:
            filepath: Path to load the state from
            state_name: Name to assign to the loaded state
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found: {filepath}")

        self.saved_states[state_name] = torch.load(filepath)
        logger.info(f"Loaded state '{state_name}' from {filepath}")

    def clear_state(self, state_name: Optional[str] = None) -> None:
        """
        Clear saved states to free memory.

        Args:
            state_name: Specific state to clear, or None to clear all
        """
        if state_name is None:
            self.saved_states.clear()
            logger.info("Cleared all saved states")
        elif state_name in self.saved_states:
            del self.saved_states[state_name]
            logger.info(f"Cleared state '{state_name}'")
        else:
            logger.warning(f"State '{state_name}' not found, nothing to clear")


def save_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Convenience function to save a model's current state.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing the model state
    """
    state_dict = {}
    for name, param in model.state_dict().items():
        if param is not None:
            state_dict[name] = param.clone().detach()
        else:
            state_dict[name] = None
    return state_dict


def restore_model_state(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Convenience function to restore a model's state.

    Args:
        model: PyTorch model
        state_dict: State dictionary to restore
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict and state_dict[name] is not None:
                param.copy_(state_dict[name])

        for name, buffer in model.named_buffers():
            if name in state_dict and state_dict[name] is not None:
                buffer.copy_(state_dict[name])


def compare_model_states(state1: Dict[str, torch.Tensor],
                         state2: Dict[str, torch.Tensor]) -> Tuple[int, int, float]:
    """
    Convenience function to compare two model states.

    Args:
        state1: First state dictionary
        state2: Second state dictionary

    Returns:
        Tuple of (changed_params, total_params, percent_changed)
    """
    total_params = 0
    changed_params = 0

    for name in state1.keys():
        if name not in state2:
            continue

        param1 = state1[name]
        param2 = state2[name]

        if param1 is None or param2 is None:
            continue

        diff = torch.abs(param1 - param2)
        num_changed = (diff > 0).sum().item()
        changed_params += num_changed
        total_params += param1.numel()

    percent_changed = 100.0 * changed_params / total_params if total_params > 0 else 0.0

    return changed_params, total_params, percent_changed
