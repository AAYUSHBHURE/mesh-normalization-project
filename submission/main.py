"""
Main execution script for 3D Mesh Normalization Project
Runs all tasks: Load, Normalize, Quantize, Reconstruct, Analyze
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mesh_loader import load_all_meshes, compute_statistics, print_statistics
from src.normalization import MinMaxNormalizer, UnitSphereNormalizer
from src.quantization import quantize_mesh, dequantize_mesh
from src.error_analysis import (compute_all_metrics, print_error_metrics,
                                 plot_error_comparison, plot_per_axis_errors,
                                 plot_error_distribution)
from src.visualization import save_mesh_file, visualize_mesh, visualize_comparison


def ensure_directories():
    """Create output directories if they don't exist."""
    directories = [
        'output/meshes',
        'output/plots',
        'output/visualizations'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def task1_load_and_inspect(data_folder='data'):
    """
    Task 1: Load and inspect meshes.
    
    Returns:
        list: Loaded meshes [(filename, vertices, faces), ...]
    """
    print("\n" + "="*70)
    print("TASK 1: LOAD AND INSPECT MESH")
    print("="*70)
    
    meshes = load_all_meshes(data_folder)
    
    if not meshes:
        print(f"\n⚠ No .obj files found in '{data_folder}/' folder.")
        print("Please add .obj mesh files to continue.")
        return []
    
    print(f"\n✓ Successfully loaded {len(meshes)} mesh(es)\n")
    
    # Print statistics for each mesh
    for filename, vertices, faces in meshes:
        stats = compute_statistics(vertices)
        print_statistics(stats, filename)
    
    return meshes


def task2_normalize_and_quantize(meshes, bins=1024):
    """
    Task 2: Normalize and quantize meshes.
    
    Args:
        meshes: List of (filename, vertices, faces) tuples
        bins: Number of quantization bins
        
    Returns:
        dict: Processed results for each mesh and method
    """
    print("\n" + "="*70)
    print("TASK 2: NORMALIZE AND QUANTIZE MESH")
    print("="*70)
    
    results = {}
    
    normalization_methods = {
        'MinMax': MinMaxNormalizer(),
        'UnitSphere': UnitSphereNormalizer()
    }
    
    for filename, vertices, faces in meshes:
        mesh_name = os.path.splitext(filename)[0]
        results[mesh_name] = {}
        
        print(f"\n{'─'*70}")
        print(f"Processing: {filename}")
        print(f"{'─'*70}")
        
        for method_name, normalizer in normalization_methods.items():
            print(f"\n→ Applying {method_name} normalization...")
            
            # Normalize and quantize
            quantized, params = quantize_mesh(vertices, normalizer, bins)
            
            # Store results
            results[mesh_name][method_name] = {
                'original_vertices': vertices,
                'faces': faces,
                'quantized': quantized,
                'params': params,
                'normalizer': normalizer
            }
            
            # Save quantized mesh
            output_path = f'output/meshes/{mesh_name}_{method_name}_quantized.ply'
            if faces is not None and len(faces) > 0:
                # For saving, we need to dequantize back to float coordinates
                dequantized_verts = dequantize_mesh(quantized, params, normalizer)
                save_mesh_file(dequantized_verts, faces, output_path)
            
            print(f"  ✓ Quantized with {bins} bins")
            print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "="*70)
    print("✓ Task 2 completed successfully!")
    print("="*70)
    
    return results


def task3_reconstruct_and_analyze(results):
    """
    Task 3: Dequantize, denormalize, and measure errors.
    
    Args:
        results: Processed results from task2
    """
    print("\n" + "="*70)
    print("TASK 3: RECONSTRUCTION AND ERROR ANALYSIS")
    print("="*70)
    
    all_error_results = {}
    
    for mesh_name, methods in results.items():
        print(f"\n{'─'*70}")
        print(f"Analyzing: {mesh_name}")
        print(f"{'─'*70}")
        
        error_results = {}
        
        for method_name, data in methods.items():
            print(f"\n→ Analyzing {method_name} reconstruction...")
            
            # Reconstruct
            original = data['original_vertices']
            quantized = data['quantized']
            params = data['params']
            normalizer = data['normalizer']
            
            reconstructed = dequantize_mesh(quantized, params, normalizer)
            
            # Compute errors
            metrics = compute_all_metrics(original, reconstructed)
            error_results[method_name] = metrics
            
            # Print metrics
            print_error_metrics(metrics, method_name)
            
            # Save reconstructed mesh
            if data['faces'] is not None and len(data['faces']) > 0:
                output_path = f'output/meshes/{mesh_name}_{method_name}_reconstructed.ply'
                save_mesh_file(reconstructed, data['faces'], output_path)
                print(f"✓ Saved reconstructed mesh: {output_path}")
            
            # Generate plots
            plot_per_axis_errors(
                metrics, 
                method_name,
                f'output/plots/{mesh_name}_{method_name}_per_axis.png'
            )
            
            plot_error_distribution(
                original,
                reconstructed,
                method_name,
                f'output/plots/{mesh_name}_{method_name}_distribution.png'
            )
        
        # Compare methods for this mesh
        if len(error_results) > 1:
            plot_error_comparison(
                error_results,
                f'output/plots/{mesh_name}_comparison.png'
            )
        
        all_error_results[mesh_name] = error_results
    
    # Print conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    for mesh_name, error_results in all_error_results.items():
        print(f"\nMesh: {mesh_name}")
        print(f"{'─'*50}")
        
        best_method = min(error_results.keys(), 
                         key=lambda m: error_results[m]['mse'])
        best_mse = error_results[best_method]['mse']
        
        print(f"Best normalization method: {best_method}")
        print(f"  → Lowest MSE: {best_mse:.8f}")
        
        print(f"\nComparison of all methods:")
        for method, metrics in error_results.items():
            print(f"  {method:<12} - MSE: {metrics['mse']:.8f}, "
                  f"MAE: {metrics['mae']:.8f}, Max: {metrics['max_error']:.8f}")
        
        print(f"\nObservations:")
        if best_method == 'MinMax':
            print("  • Min-Max normalization provides better reconstruction accuracy.")
            print("  • This is expected as it preserves the original coordinate ranges.")
        else:
            print("  • Unit Sphere normalization provides better reconstruction accuracy.")
            print("  • This suggests the mesh benefits from centered, radial scaling.")
    
    print("\n" + "="*70)
    print("✓ Task 3 completed successfully!")
    print("="*70)
    
    return all_error_results


def generate_summary_report(results, error_results):
    """
    Generate a summary report of all results.
    
    Args:
        results: Results from task2
        error_results: Results from task3
    """
    report_path = 'output/SUMMARY_REPORT.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("3D MESH NORMALIZATION - SUMMARY REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for mesh_name in results.keys():
            f.write(f"\nMesh: {mesh_name}\n")
            f.write("─"*70 + "\n")
            
            mesh_errors = error_results.get(mesh_name, {})
            
            for method_name, metrics in mesh_errors.items():
                f.write(f"\n{method_name} Normalization:\n")
                f.write(f"  MSE:        {metrics['mse']:.8f}\n")
                f.write(f"  MAE:        {metrics['mae']:.8f}\n")
                f.write(f"  RMSE:       {metrics['rmse']:.8f}\n")
                f.write(f"  Max Error:  {metrics['max_error']:.8f}\n")
                
                f.write(f"\n  Per-Axis Errors:\n")
                for axis in ['x', 'y', 'z']:
                    ax_data = metrics['per_axis'][axis]
                    f.write(f"    {axis.upper()}: MSE={ax_data['mse']:.8f}, "
                           f"MAE={ax_data['mae']:.8f}\n")
            
            if mesh_errors:
                best = min(mesh_errors.keys(), key=lambda m: mesh_errors[m]['mse'])
                f.write(f"\n  ✓ Best Method: {best}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ Summary report saved: {report_path}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" 3D MESH NORMALIZATION, QUANTIZATION, AND ERROR ANALYSIS")
    print("="*70)
    print("\nThis program implements:")
    print("  1. Mesh loading and inspection")
    print("  2. Normalization (Min-Max and Unit Sphere)")
    print("  3. Quantization (1024 bins)")
    print("  4. Reconstruction and error analysis")
    print("="*70)
    
    # Ensure output directories exist
    ensure_directories()
    
    # Task 1: Load meshes
    meshes = task1_load_and_inspect('data')
    
    if not meshes:
        print("\n⚠ No meshes to process. Exiting.")
        return
    
    # Task 2: Normalize and quantize
    results = task2_normalize_and_quantize(meshes, bins=1024)
    
    # Task 3: Reconstruct and analyze
    error_results = task3_reconstruct_and_analyze(results)
    
    # Generate summary
    generate_summary_report(results, error_results)
    
    print("\n" + "="*70)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nOutput files generated:")
    print("  • Meshes:         output/meshes/")
    print("  • Plots:          output/plots/")
    print("  • Summary:        output/SUMMARY_REPORT.txt")
    print("\nYou can now review the results and create your final PDF report.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
