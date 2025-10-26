"""
Result Manager for PINN-PBM.

Handles timestamped saving and loading of experiment results including
model weights, loss histories, predictions, plots, and metadata.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import tensorflow as tf


class ResultManager:
    """Manages timestamped saving and loading of PINN experiment results.
    
    This class provides a clean interface for saving all artifacts from a PINN
    training run into timestamped directories, preventing result overwrites and
    enabling easy experiment tracking.
    
    Directory structure:
        results/{problem_type}/{case_name}/{timestamp}/
            ├── model_weights.weights.h5
            ├── model_weights_amt.weights.h5 (if applicable)
            ├── loss_history.npz
            ├── predictions.npz
            ├── metadata.json
            └── plots/
                ├── comparison_plot.png
                └── loss_history_plot.png
    
    Attributes:
        base_dir: Root directory for all results
        problem_type: Type of problem (e.g., 'breakage', 'aggregation')
        case_name: Name of the specific case (e.g., 'case1_linear')
        timestamp: Current timestamp for this run
        save_dir: Full path to the timestamped save directory
        
    Example:
        >>> rm = ResultManager(problem_type='breakage', case_name='case1_linear')
        >>> # After training
        >>> rm.save_model(pinn.model, pinn.amt)
        >>> rm.save_loss_history(pinn.train_loss_history, 
        ...                       pinn.data_loss_history,
        ...                       pinn.physics_loss_history)
        >>> rm.save_predictions(v_plot, t_plot, f_pred, f_exact)
        >>> print(f"Results saved to: {rm.save_dir}")
    """
    
    def __init__(
        self,
        problem_type: str,
        case_name: str,
        base_dir: str = "results",
        timestamp: Optional[str] = None,
        create_dir: bool = True
    ):
        """Initialize the ResultManager.
        
        Args:
            problem_type: Type of problem (e.g., 'breakage', 'aggregation')
            case_name: Name of the case (e.g., 'case1_linear', 'case2_quadratic')
            base_dir: Base directory for all results (default: 'results')
            timestamp: Custom timestamp string (default: auto-generated YYYYMMDD_HHMMSS)
            create_dir: Whether to create the directory immediately (default: True)
            
        Example:
            >>> # Auto-generated timestamp
            >>> rm = ResultManager('breakage', 'case1_linear')
            >>> 
            >>> # Custom timestamp (for loading existing results)
            >>> rm = ResultManager('breakage', 'case1_linear', 
            ...                    timestamp='20241027_013045',
            ...                    create_dir=False)
        """
        self.base_dir = base_dir
        self.problem_type = problem_type
        self.case_name = case_name
        
        # Generate or use provided timestamp
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = timestamp
            
        # Construct full save directory path
        self.save_dir = os.path.join(
            self.base_dir,
            self.problem_type,
            self.case_name,
            self.timestamp
        )
        
        # Create directory if requested
        if create_dir:
            self._create_directories()
    
    def _create_directories(self) -> None:
        """Create the directory structure for saving results.
        
        Creates:
            - Main save directory
            - plots/ subdirectory for figures
        """
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "plots"), exist_ok=True)
    
    def save_model(
        self,
        main_model: tf.keras.Model,
        aux_model: Optional[tf.keras.Model] = None,
        main_name: str = "model_weights.weights.h5",
        aux_name: str = "model_weights_amt.weights.h5"
    ) -> Tuple[str, Optional[str]]:
        """Save TensorFlow/Keras model weights.
        
        Args:
            main_model: Primary PINN model to save
            aux_model: Auxiliary model (e.g., AMT head for delta peaks), optional
            main_name: Filename for main model weights
            aux_name: Filename for auxiliary model weights
            
        Returns:
            Tuple of (main_model_path, aux_model_path)
            aux_model_path is None if no auxiliary model provided
            
        Example:
            >>> rm.save_model(pinn.model, pinn.amt)
            ('/path/to/results/.../model_weights.weights.h5',
             '/path/to/results/.../model_weights_amt.weights.h5')
        """
        main_path = os.path.join(self.save_dir, main_name)
        main_model.save_weights(main_path)
        
        aux_path = None
        if aux_model is not None:
            aux_path = os.path.join(self.save_dir, aux_name)
            aux_model.save_weights(aux_path)
        
        return main_path, aux_path
    
    def load_model(
        self,
        main_model: tf.keras.Model,
        aux_model: Optional[tf.keras.Model] = None,
        main_name: str = "model_weights.weights.h5",
        aux_name: str = "model_weights_amt.weights.h5"
    ) -> None:
        """Load TensorFlow/Keras model weights.
        
        Args:
            main_model: Primary PINN model (must have same architecture)
            aux_model: Auxiliary model (must have same architecture), optional
            main_name: Filename of main model weights
            aux_name: Filename of auxiliary model weights
            
        Raises:
            FileNotFoundError: If weight files don't exist
            
        Example:
            >>> # Create model with same architecture
            >>> pinn = BreakagePINN(...)
            >>> rm = ResultManager('breakage', 'case1_linear', 
            ...                    timestamp='20241027_013045',
            ...                    create_dir=False)
            >>> rm.load_model(pinn.model, pinn.amt)
        """
        main_path = os.path.join(self.save_dir, main_name)
        if not os.path.exists(main_path):
            raise FileNotFoundError(f"Model weights not found: {main_path}")
        main_model.load_weights(main_path)
        
        if aux_model is not None:
            aux_path = os.path.join(self.save_dir, aux_name)
            if not os.path.exists(aux_path):
                raise FileNotFoundError(f"Auxiliary model weights not found: {aux_path}")
            aux_model.load_weights(aux_path)
    
    def save_loss_history(
        self,
        train_loss: List[float],
        data_loss: List[float],
        physics_loss: List[float],
        filename: str = "loss_history.npz"
    ) -> str:
        """Save training loss histories.
        
        Args:
            train_loss: Total loss history over training
            data_loss: Data loss history
            physics_loss: Physics loss history
            filename: Name of file to save (default: 'loss_history.npz')
            
        Returns:
            Full path to saved file
            
        Example:
            >>> rm.save_loss_history(
            ...     pinn.train_loss_history,
            ...     pinn.data_loss_history,
            ...     pinn.physics_loss_history
            ... )
        """
        filepath = os.path.join(self.save_dir, filename)
        np.savez(
            filepath,
            train_loss=np.array(train_loss),
            data_loss=np.array(data_loss),
            physics_loss=np.array(physics_loss)
        )
        return filepath
    
    def load_loss_history(
        self,
        filename: str = "loss_history.npz"
    ) -> Dict[str, np.ndarray]:
        """Load training loss histories.
        
        Args:
            filename: Name of file to load (default: 'loss_history.npz')
            
        Returns:
            Dictionary with keys 'train_loss', 'data_loss', 'physics_loss'
            
        Raises:
            FileNotFoundError: If loss history file doesn't exist
            
        Example:
            >>> losses = rm.load_loss_history()
            >>> train_loss = losses['train_loss']
            >>> data_loss = losses['data_loss']
        """
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Loss history not found: {filepath}")
        
        data = np.load(filepath)
        return {
            'train_loss': data['train_loss'],
            'data_loss': data['data_loss'],
            'physics_loss': data['physics_loss']
        }
    
    def save_predictions(
        self,
        v_points: np.ndarray,
        t_points: np.ndarray,
        predictions: np.ndarray,
        exact_solutions: Optional[np.ndarray] = None,
        filename: str = "predictions.npz"
    ) -> str:
        """Save PINN predictions and (optionally) exact solutions.
        
        Args:
            v_points: Volume grid points (1D array)
            t_points: Time grid points (1D array)
            predictions: PINN predictions (2D array: [n_t, n_v])
            exact_solutions: Exact analytical solutions (2D array), optional
            filename: Name of file to save (default: 'predictions.npz')
            
        Returns:
            Full path to saved file
            
        Example:
            >>> rm.save_predictions(v_plot, t_plot, f_pred, f_exact)
        """
        filepath = os.path.join(self.save_dir, filename)
        
        save_dict = {
            'v_points': v_points,
            't_points': t_points,
            'predictions': predictions
        }
        
        if exact_solutions is not None:
            save_dict['exact_solutions'] = exact_solutions
        
        np.savez(filepath, **save_dict)
        return filepath
    
    def load_predictions(
        self,
        filename: str = "predictions.npz"
    ) -> Dict[str, np.ndarray]:
        """Load PINN predictions and grids.
        
        Args:
            filename: Name of file to load (default: 'predictions.npz')
            
        Returns:
            Dictionary with keys 'v_points', 't_points', 'predictions',
            and optionally 'exact_solutions'
            
        Raises:
            FileNotFoundError: If predictions file doesn't exist
            
        Example:
            >>> data = rm.load_predictions()
            >>> v_plot = data['v_points']
            >>> predictions = data['predictions']
        """
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Predictions not found: {filepath}")
        
        data = np.load(filepath)
        result = {
            'v_points': data['v_points'],
            't_points': data['t_points'],
            'predictions': data['predictions']
        }
        
        if 'exact_solutions' in data:
            result['exact_solutions'] = data['exact_solutions']
        
        return result
    
    def save_metadata(
        self,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        filename: str = "metadata.json"
    ) -> str:
        """Save experiment metadata and configuration.
        
        Args:
            config: Configuration dictionary (hyperparameters, settings, etc.)
            metrics: Final metrics dictionary (losses, errors, etc.), optional
            filename: Name of file to save (default: 'metadata.json')
            
        Returns:
            Full path to saved file
            
        Example:
            >>> config = {
            ...     'case_type': 'case1',
            ...     'v_min': 0.001, 'v_max': 10.0,
            ...     'epochs': 3000, 'learning_rate': 0.0005
            ... }
            >>> metrics = {
            ...     'final_train_loss': 1.23e-4,
            ...     'final_data_loss': 5.67e-5,
            ...     'mean_relative_error': 0.012
            ... }
            >>> rm.save_metadata(config, metrics)
        """
        filepath = os.path.join(self.save_dir, filename)
        
        metadata = {
            'timestamp': self.timestamp,
            'problem_type': self.problem_type,
            'case_name': self.case_name,
            'config': config
        }
        
        if metrics is not None:
            metadata['metrics'] = metrics
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def load_metadata(
        self,
        filename: str = "metadata.json"
    ) -> Dict[str, Any]:
        """Load experiment metadata.
        
        Args:
            filename: Name of file to load (default: 'metadata.json')
            
        Returns:
            Dictionary containing metadata, config, and optionally metrics
            
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            
        Example:
            >>> metadata = rm.load_metadata()
            >>> config = metadata['config']
            >>> metrics = metadata.get('metrics', {})
        """
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metadata not found: {filepath}")
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_plot_path(self, plot_name: str) -> str:
        """Get the full path for saving a plot.
        
        Args:
            plot_name: Name of the plot file (e.g., 'comparison.png')
            
        Returns:
            Full path to save the plot
            
        Example:
            >>> plot_path = rm.get_plot_path('comparison.png')
            >>> plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        """
        return os.path.join(self.save_dir, "plots", plot_name)
    
    def list_saved_files(self) -> List[str]:
        """List all files saved in this run's directory.
        
        Returns:
            List of relative file paths within the save directory
            
        Example:
            >>> files = rm.list_saved_files()
            >>> print("Saved files:")
            >>> for f in files:
            ...     print(f"  - {f}")
        """
        all_files = []
        for root, _, files in os.walk(self.save_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.save_dir)
                all_files.append(rel_path)
        return sorted(all_files)
    
    @staticmethod
    def list_all_runs(
        problem_type: str,
        case_name: str,
        base_dir: str = "results"
    ) -> List[str]:
        """List all saved runs for a given problem type and case.
        
        Args:
            problem_type: Type of problem (e.g., 'breakage')
            case_name: Name of the case (e.g., 'case1_linear')
            base_dir: Base directory for results (default: 'results')
            
        Returns:
            List of timestamp strings for all available runs
            
        Example:
            >>> runs = ResultManager.list_all_runs('breakage', 'case1_linear')
            >>> print(f"Found {len(runs)} previous runs:")
            >>> for run in runs:
            ...     print(f"  - {run}")
        """
        case_dir = os.path.join(base_dir, problem_type, case_name)
        
        if not os.path.exists(case_dir):
            return []
        
        timestamps = []
        for item in os.listdir(case_dir):
            item_path = os.path.join(case_dir, item)
            if os.path.isdir(item_path):
                timestamps.append(item)
        
        return sorted(timestamps, reverse=True)  # Most recent first
    
    def __repr__(self) -> str:
        """String representation of ResultManager."""
        return (f"ResultManager(problem_type='{self.problem_type}', "
                f"case_name='{self.case_name}', timestamp='{self.timestamp}')")
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"ResultManager for {self.problem_type}/{self.case_name} [{self.timestamp}]"
