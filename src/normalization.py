"""
Task 2: Normalization Methods
Implements Min-Max and Unit Sphere normalization.
"""

import numpy as np


class Normalizer:
    """Base class for normalization methods."""
    
    def normalize(self, vertices):
        """
        Normalize vertices.
        
        Args:
            vertices (np.ndarray): Nx3 array of vertex coordinates
            
        Returns:
            tuple: (normalized_vertices, normalization_parameters)
        """
        raise NotImplementedError("Subclasses must implement normalize()")
    
    def denormalize(self, vertices, params):
        """
        Reverse normalization.
        
        Args:
            vertices (np.ndarray): Normalized vertices
            params (dict): Normalization parameters
            
        Returns:
            np.ndarray: Denormalized vertices
        """
        raise NotImplementedError("Subclasses must implement denormalize()")


class MinMaxNormalizer(Normalizer):
    """
    Min-Max Normalization: Scale to [0, 1] range.
    Formula: x' = (x - x_min) / (x_max - x_min)
    """
    
    def normalize(self, vertices):
        """
        Apply Min-Max normalization.
        
        Args:
            vertices (np.ndarray): Nx3 array of vertex coordinates
            
        Returns:
            tuple: (normalized_vertices, params)
                - normalized_vertices: Vertices scaled to [0, 1]
                - params: dict with 'min' and 'max' values
        """
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        
        # Avoid division by zero
        range_vals = v_max - v_min
        range_vals[range_vals == 0] = 1.0
        
        normalized = (vertices - v_min) / range_vals
        
        params = {
            'min': v_min,
            'max': v_max,
            'range': range_vals,
            'method': 'minmax'
        }
        
        return normalized, params
    
    def denormalize(self, vertices, params):
        """
        Reverse Min-Max normalization.
        
        Args:
            vertices (np.ndarray): Normalized vertices
            params (dict): Normalization parameters
            
        Returns:
            np.ndarray: Original scale vertices
        """
        return vertices * params['range'] + params['min']


class UnitSphereNormalizer(Normalizer):
    """
    Unit Sphere Normalization: Scale mesh to fit in a sphere of radius 1.
    Centers the mesh at origin and scales to unit radius.
    """
    
    def normalize(self, vertices):
        """
        Apply Unit Sphere normalization.
        
        Args:
            vertices (np.ndarray): Nx3 array of vertex coordinates
            
        Returns:
            tuple: (normalized_vertices, params)
                - normalized_vertices: Vertices in unit sphere
                - params: dict with 'center' and 'scale' values
        """
        # Center the mesh at origin
        center = vertices.mean(axis=0)
        centered = vertices - center
        
        # Find the maximum distance from center
        distances = np.linalg.norm(centered, axis=1)
        max_distance = distances.max()
        
        # Avoid division by zero
        if max_distance == 0:
            max_distance = 1.0
        
        # Scale to unit sphere
        normalized = centered / max_distance
        
        params = {
            'center': center,
            'scale': max_distance,
            'method': 'unit_sphere'
        }
        
        return normalized, params
    
    def denormalize(self, vertices, params):
        """
        Reverse Unit Sphere normalization.
        
        Args:
            vertices (np.ndarray): Normalized vertices
            params (dict): Normalization parameters
            
        Returns:
            np.ndarray: Original scale vertices
        """
        return vertices * params['scale'] + params['center']


class ZScoreNormalizer(Normalizer):
    """
    Z-Score Normalization: Center and scale by standard deviation.
    Formula: x' = (x - μ) / σ
    """
    
    def normalize(self, vertices):
        """
        Apply Z-Score normalization.
        
        Args:
            vertices (np.ndarray): Nx3 array of vertex coordinates
            
        Returns:
            tuple: (normalized_vertices, params)
        """
        mean = vertices.mean(axis=0)
        std = vertices.std(axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        normalized = (vertices - mean) / std
        
        params = {
            'mean': mean,
            'std': std,
            'method': 'zscore'
        }
        
        return normalized, params
    
    def denormalize(self, vertices, params):
        """
        Reverse Z-Score normalization.
        
        Args:
            vertices (np.ndarray): Normalized vertices
            params (dict): Normalization parameters
            
        Returns:
            np.ndarray: Original scale vertices
        """
        return vertices * params['std'] + params['mean']


def get_normalizer(method='minmax'):
    """
    Factory function to get a normalizer instance.
    
    Args:
        method (str): 'minmax', 'unit_sphere', or 'zscore'
        
    Returns:
        Normalizer: Instance of the requested normalizer
    """
    normalizers = {
        'minmax': MinMaxNormalizer,
        'unit_sphere': UnitSphereNormalizer,
        'zscore': ZScoreNormalizer
    }
    
    if method not in normalizers:
        raise ValueError(f"Unknown normalization method: {method}. "
                        f"Choose from {list(normalizers.keys())}")
    
    return normalizers[method]()


if __name__ == "__main__":
    # Test normalization methods
    print("Testing Normalization Methods\n")
    
    # Create sample vertices
    vertices = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    print("Original vertices:")
    print(vertices)
    
    # Test Min-Max
    print("\n--- Min-Max Normalization ---")
    minmax = MinMaxNormalizer()
    normalized, params = minmax.normalize(vertices)
    print("Normalized:", normalized)
    denormalized = minmax.denormalize(normalized, params)
    print("Denormalized:", denormalized)
    print("Error:", np.abs(vertices - denormalized).max())
    
    # Test Unit Sphere
    print("\n--- Unit Sphere Normalization ---")
    sphere = UnitSphereNormalizer()
    normalized, params = sphere.normalize(vertices)
    print("Normalized:", normalized)
    denormalized = sphere.denormalize(normalized, params)
    print("Denormalized:", denormalized)
    print("Error:", np.abs(vertices - denormalized).max())
