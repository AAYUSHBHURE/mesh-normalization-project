"""
Bonus Task: Advanced Features
This module contains optional bonus task implementations.
"""

import numpy as np
from src.normalization import Normalizer
from src.quantization import quantize_vertices, dequantize_vertices


class RotationInvariantNormalizer(Normalizer):
    """
    Bonus Option 2: Rotation and Translation Invariant Normalization
    
    This normalizer centers the mesh at the origin and aligns it
    to principal components, making it invariant to rotation.
    """
    
    def normalize(self, vertices):
        """
        Apply rotation-invariant normalization using PCA.
        
        Steps:
        1. Center mesh at origin
        2. Compute principal components (PCA)
        3. Align to principal axes
        4. Scale to unit sphere
        """
        # Center at origin
        center = vertices.mean(axis=0)
        centered = vertices - center
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Compute eigenvectors (principal components)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Align to principal axes
        aligned = centered @ eigenvectors
        
        # Scale to unit sphere
        max_distance = np.linalg.norm(aligned, axis=1).max()
        if max_distance == 0:
            max_distance = 1.0
        
        normalized = aligned / max_distance
        
        params = {
            'center': center,
            'eigenvectors': eigenvectors,
            'scale': max_distance,
            'method': 'rotation_invariant'
        }
        
        return normalized, params
    
    def denormalize(self, vertices, params):
        """Reverse rotation-invariant normalization."""
        # Unscale
        unscaled = vertices * params['scale']
        
        # Rotate back to original orientation
        original_orientation = unscaled @ params['eigenvectors'].T
        
        # Translate back
        denormalized = original_orientation + params['center']
        
        return denormalized


def adaptive_quantization(vertices, target_bins=1024, density_factor=2.0):
    """
    Bonus Option 2: Adaptive Quantization
    
    Allocates more bins to dense regions and fewer to sparse regions.
    
    Args:
        vertices (np.ndarray): Normalized vertices [0, 1]
        target_bins (int): Total number of bins to use
        density_factor (float): How much to favor dense regions
        
    Returns:
        np.ndarray: Adaptively quantized vertices
    """
    # Compute local density using histogram
    hist_bins = 32  # Use coarser bins to estimate density
    
    quantized = np.zeros_like(vertices, dtype=np.int32)
    
    for axis in range(3):
        axis_data = vertices[:, axis]
        
        # Compute density histogram
        density, edges = np.histogram(axis_data, bins=hist_bins, range=(0, 1))
        
        # Compute adaptive bin allocation
        # More bins to dense regions
        density_norm = density / density.sum()
        bins_per_region = np.maximum(
            1, 
            (density_norm * target_bins * density_factor).astype(int)
        )
        
        # Normalize to target bins
        bins_per_region = (bins_per_region * target_bins / bins_per_region.sum()).astype(int)
        
        # Create adaptive edges
        adaptive_edges = [0.0]
        for i, num_bins in enumerate(bins_per_region):
            region_start = edges[i]
            region_end = edges[i + 1]
            region_edges = np.linspace(region_start, region_end, num_bins + 1)[1:]
            adaptive_edges.extend(region_edges)
        
        # Quantize using adaptive edges
        quantized[:, axis] = np.digitize(axis_data, adaptive_edges) - 1
        quantized[:, axis] = np.clip(quantized[:, axis], 0, len(adaptive_edges) - 1)
    
    return quantized


def generate_rotated_versions(vertices, num_rotations=5):
    """
    Generate randomly rotated versions of a mesh.
    
    Args:
        vertices (np.ndarray): Original vertices
        num_rotations (int): Number of rotated versions to generate
        
    Returns:
        list: List of rotated vertex arrays
    """
    rotated_meshes = []
    
    for _ in range(num_rotations):
        # Generate random rotation angles
        angles = np.random.uniform(0, 2*np.pi, 3)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Apply rotation
        rotated = vertices @ R.T
        
        # Add random translation
        translation = np.random.uniform(-1, 1, 3)
        rotated += translation
        
        rotated_meshes.append(rotated)
    
    return rotated_meshes


if __name__ == "__main__":
    print("Bonus Task Module - Advanced Features\n")
    
    # Test rotation-invariant normalization
    print("Testing Rotation-Invariant Normalization...")
    test_vertices = np.random.rand(100, 3) * 10
    
    normalizer = RotationInvariantNormalizer()
    normalized, params = normalizer.normalize(test_vertices)
    denormalized = normalizer.denormalize(normalized, params)
    
    error = np.abs(test_vertices - denormalized).max()
    print(f"Reconstruction error: {error:.8f}")
    
    # Test with rotated version
    rotated_versions = generate_rotated_versions(test_vertices, 3)
    print(f"\nGenerated {len(rotated_versions)} rotated versions")
    
    print("\n✓ Bonus task module working!")
