"""
Task 1: Load and Inspect 3D Mesh
This module handles loading .obj files and computing statistics.
"""

import trimesh
import numpy as np
import os


def load_mesh(filepath):
    """
    Load a mesh file and extract vertices and faces.
    
    Args:
        filepath (str): Path to the .obj file
        
    Returns:
        tuple: (vertices, faces) as numpy arrays
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath}")
    
    mesh = trimesh.load(filepath)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces) if hasattr(mesh, 'faces') else None
    
    return vertices, faces


def compute_statistics(vertices):
    """
    Compute and return basic statistics for mesh vertices.
    
    Args:
        vertices (np.ndarray): Nx3 array of vertex coordinates
        
    Returns:
        dict: Statistics including count, min, max, mean, std per axis
    """
    stats = {
        'num_vertices': len(vertices),
        'min': vertices.min(axis=0),
        'max': vertices.max(axis=0),
        'mean': vertices.mean(axis=0),
        'std': vertices.std(axis=0),
        'range': vertices.max(axis=0) - vertices.min(axis=0)
    }
    return stats


def print_statistics(stats, mesh_name="Mesh"):
    """
    Print mesh statistics in a readable format.
    
    Args:
        stats (dict): Statistics dictionary from compute_statistics
        mesh_name (str): Name of the mesh for display
    """
    print(f"\n{'='*60}")
    print(f"Statistics for: {mesh_name}")
    print(f"{'='*60}")
    print(f"Number of vertices: {stats['num_vertices']}")
    print(f"\nPer-axis statistics:")
    print(f"{'Axis':<10} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std Dev':<15}")
    print(f"{'-'*70}")
    
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"{axis:<10} {stats['min'][i]:<15.6f} {stats['max'][i]:<15.6f} "
              f"{stats['mean'][i]:<15.6f} {stats['std'][i]:<15.6f}")
    
    print(f"\nRange (max - min):")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"  {axis}: {stats['range'][i]:.6f}")
    print(f"{'='*60}\n")


def load_all_meshes(data_folder):
    """
    Load all .obj files from a folder.
    
    Args:
        data_folder (str): Path to folder containing .obj files
        
    Returns:
        list: List of tuples (filename, vertices, faces)
    """
    meshes = []
    
    if not os.path.exists(data_folder):
        print(f"Warning: Data folder '{data_folder}' does not exist.")
        return meshes
    
    for filename in os.listdir(data_folder):
        if filename.endswith('.obj'):
            filepath = os.path.join(data_folder, filename)
            try:
                vertices, faces = load_mesh(filepath)
                meshes.append((filename, vertices, faces))
                print(f"✓ Loaded: {filename} ({len(vertices)} vertices)")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
    
    return meshes


if __name__ == "__main__":
    # Test the mesh loader
    print("Mesh Loader Module - Test")
    print("Place .obj files in the 'data/' folder to test this module.")
    
    # Try to load meshes from data folder
    meshes = load_all_meshes('../data')
    
    if meshes:
        filename, vertices, faces = meshes[0]
        stats = compute_statistics(vertices)
        print_statistics(stats, filename)
    else:
        print("\nNo mesh files found. Please add .obj files to the data/ folder.")
