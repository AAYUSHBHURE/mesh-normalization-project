"""
Task 3: Error Analysis and Metrics
Compute and analyze reconstruction errors.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def compute_mse(original, reconstructed):
    """
    Compute Mean Squared Error.
    
    Args:
        original (np.ndarray): Original vertices
        reconstructed (np.ndarray): Reconstructed vertices
        
    Returns:
        float: MSE value
    """
    return np.mean((original - reconstructed) ** 2)


def compute_mae(original, reconstructed):
    """
    Compute Mean Absolute Error.
    
    Args:
        original (np.ndarray): Original vertices
        reconstructed (np.ndarray): Reconstructed vertices
        
    Returns:
        float: MAE value
    """
    return np.mean(np.abs(original - reconstructed))


def compute_per_axis_errors(original, reconstructed):
    """
    Compute errors separately for each axis (X, Y, Z).
    
    Args:
        original (np.ndarray): Original vertices (Nx3)
        reconstructed (np.ndarray): Reconstructed vertices (Nx3)
        
    Returns:
        dict: Errors per axis
    """
    errors = {
        'x': {},
        'y': {},
        'z': {}
    }
    
    axes = ['x', 'y', 'z']
    for i, axis in enumerate(axes):
        orig_axis = original[:, i]
        recon_axis = reconstructed[:, i]
        
        errors[axis]['mse'] = np.mean((orig_axis - recon_axis) ** 2)
        errors[axis]['mae'] = np.mean(np.abs(orig_axis - recon_axis))
        errors[axis]['max_error'] = np.max(np.abs(orig_axis - recon_axis))
        errors[axis]['std'] = np.std(orig_axis - recon_axis)
    
    return errors


def compute_all_metrics(original, reconstructed):
    """
    Compute comprehensive error metrics.
    
    Args:
        original (np.ndarray): Original vertices
        reconstructed (np.ndarray): Reconstructed vertices
        
    Returns:
        dict: All error metrics
    """
    metrics = {
        'mse': compute_mse(original, reconstructed),
        'mae': compute_mae(original, reconstructed),
        'rmse': np.sqrt(compute_mse(original, reconstructed)),
        'max_error': np.max(np.abs(original - reconstructed)),
        'per_axis': compute_per_axis_errors(original, reconstructed)
    }
    
    return metrics


def print_error_metrics(metrics, method_name="Method"):
    """
    Print error metrics in a readable format.
    
    Args:
        metrics (dict): Error metrics dictionary
        method_name (str): Name of the normalization method
    """
    print(f"\n{'='*60}")
    print(f"Error Metrics for: {method_name}")
    print(f"{'='*60}")
    print(f"Mean Squared Error (MSE):  {metrics['mse']:.8f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.8f}")
    print(f"Root MSE (RMSE):           {metrics['rmse']:.8f}")
    print(f"Maximum Error:             {metrics['max_error']:.8f}")
    
    print(f"\nPer-Axis Errors:")
    print(f"{'Axis':<6} {'MSE':<15} {'MAE':<15} {'Max Error':<15} {'Std Dev':<15}")
    print(f"{'-'*66}")
    
    for axis in ['x', 'y', 'z']:
        ax_data = metrics['per_axis'][axis]
        print(f"{axis.upper():<6} {ax_data['mse']:<15.8f} {ax_data['mae']:<15.8f} "
              f"{ax_data['max_error']:<15.8f} {ax_data['std']:<15.8f}")
    
    print(f"{'='*60}\n")


def plot_error_comparison(error_results, output_path='output/plots/error_comparison.png'):
    """
    Plot error comparison between different normalization methods.
    
    Args:
        error_results (dict): Dict mapping method names to their metrics
        output_path (str): Path to save the plot
    """
    methods = list(error_results.keys())
    
    # Extract metrics
    mse_values = [error_results[m]['mse'] for m in methods]
    mae_values = [error_results[m]['mae'] for m in methods]
    max_errors = [error_results[m]['max_error'] for m in methods]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE comparison
    axes[0].bar(methods, mse_values, color='steelblue')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Mean Squared Error Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    axes[1].bar(methods, mae_values, color='coral')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Max Error comparison
    axes[2].bar(methods, max_errors, color='mediumseagreen')
    axes[2].set_ylabel('Max Error')
    axes[2].set_title('Maximum Error Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error comparison plot: {output_path}")
    plt.close()


def plot_per_axis_errors(metrics, method_name, output_path='output/plots/per_axis_error.png'):
    """
    Plot per-axis error breakdown.
    
    Args:
        metrics (dict): Error metrics for a single method
        method_name (str): Name of the method
        output_path (str): Path to save the plot
    """
    axes_names = ['X', 'Y', 'Z']
    mse_values = [metrics['per_axis']['x']['mse'], 
                  metrics['per_axis']['y']['mse'], 
                  metrics['per_axis']['z']['mse']]
    mae_values = [metrics['per_axis']['x']['mae'], 
                  metrics['per_axis']['y']['mae'], 
                  metrics['per_axis']['z']['mae']]
    
    x = np.arange(len(axes_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, mse_values, width, label='MSE', color='steelblue')
    ax.bar(x + width/2, mae_values, width, label='MAE', color='coral')
    
    ax.set_xlabel('Axis')
    ax.set_ylabel('Error')
    ax.set_title(f'Per-Axis Error Analysis - {method_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(axes_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-axis error plot: {output_path}")
    plt.close()


def plot_error_distribution(original, reconstructed, method_name, 
                           output_path='output/plots/error_distribution.png'):
    """
    Plot distribution of reconstruction errors.
    
    Args:
        original (np.ndarray): Original vertices
        reconstructed (np.ndarray): Reconstructed vertices
        method_name (str): Name of the method
        output_path (str): Path to save the plot
    """
    errors = np.abs(original - reconstructed)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        axes[i].hist(errors[:, i], bins=50, color=['steelblue', 'coral', 'mediumseagreen'][i], 
                    alpha=0.7, edgecolor='black')
        axes[i].set_xlabel('Absolute Error')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{axis_name}-axis Error Distribution')
        axes[i].grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'Error Distribution - {method_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error distribution plot: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Test error analysis
    print("Testing Error Analysis\n")
    
    # Create sample data
    original = np.random.rand(100, 3) * 10
    # Add some noise to simulate reconstruction error
    reconstructed = original + np.random.randn(100, 3) * 0.01
    
    # Compute metrics
    metrics = compute_all_metrics(original, reconstructed)
    print_error_metrics(metrics, "Test Method")
    
    # Test plotting
    plot_per_axis_errors(metrics, "Test Method", 'test_per_axis.png')
    plot_error_distribution(original, reconstructed, "Test Method", 'test_distribution.png')
    
    print("\nTest plots generated successfully!")
