"""
Task 2: Quantization and Dequantization
Converts continuous values to discrete bins and back.
"""

import numpy as np


def quantize_vertices(vertices, bins=1024):
    """
    Quantize normalized vertices to discrete bins.
    
    Note: Vertices should be normalized to [0, 1] range before quantization.
    
    Args:
        vertices (np.ndarray): Nx3 array of normalized vertices (in [0, 1] range)
        bins (int): Number of bins (default: 1024)
        
    Returns:
        np.ndarray: Quantized vertices as integers [0, bins-1]
    """
    # Ensure vertices are in [0, 1] range
    vertices_clipped = np.clip(vertices, 0.0, 1.0)
    
    # Quantize: q = round(x * (bins - 1)) for unbiased binning (Draco-style)
    quantized = np.rint(vertices_clipped * (bins - 1)).astype(np.int32)
    
    # Ensure values are within valid range
    quantized = np.clip(quantized, 0, bins - 1)
    
    return quantized


def dequantize_vertices(quantized_vertices, bins=1024):
    """
    Dequantize discrete bins back to continuous values.
    
    Args:
        quantized_vertices (np.ndarray): Nx3 array of quantized vertices (integers)
        bins (int): Number of bins used in quantization (default: 1024)
        
    Returns:
        np.ndarray: Dequantized vertices in [0, 1] range
    """
    # Dequantize: x = q / (bins - 1)
    dequantized = quantized_vertices.astype(np.float64) / (bins - 1)
    
    return dequantized


def quantize_mesh(vertices, normalizer, bins=1024):
    """
    Complete quantization pipeline: normalize then quantize.
    
    Args:
        vertices (np.ndarray): Original mesh vertices
        normalizer: Normalizer instance (MinMaxNormalizer or UnitSphereNormalizer)
        bins (int): Number of quantization bins
        
    Returns:
        tuple: (quantized_vertices, normalization_params)
    """
    # Normalize
    normalized, params = normalizer.normalize(vertices)
    
    # For non-MinMax normalizers, we need to map to [0, 1] for quantization
    if params['method'] != 'minmax':
        # Map to [0, 1] range for quantization
        v_min = normalized.min(axis=0)
        v_max = normalized.max(axis=0)
        range_vals = v_max - v_min
        range_vals[range_vals == 0] = 1.0
        normalized_01 = (normalized - v_min) / range_vals
        
        # Store additional mapping parameters
        params['quant_min'] = v_min
        params['quant_max'] = v_max
        params['quant_range'] = range_vals
    else:
        normalized_01 = normalized
    
    # Quantize
    quantized = quantize_vertices(normalized_01, bins)
    
    params['bins'] = bins
    
    return quantized, params


def dequantize_mesh(quantized_vertices, params, normalizer):
    """
    Complete dequantization pipeline: dequantize then denormalize.
    
    Args:
        quantized_vertices (np.ndarray): Quantized mesh vertices
        params (dict): Normalization and quantization parameters
        normalizer: Normalizer instance used for original normalization
        
    Returns:
        np.ndarray: Reconstructed vertices in original scale
    """
    bins = params.get('bins', 1024)
    
    # Dequantize to [0, 1]
    dequantized = dequantize_vertices(quantized_vertices, bins)
    
    # If we had mapped to [0, 1] for quantization, reverse that mapping
    if params['method'] != 'minmax' and 'quant_min' in params:
        dequantized = dequantized * params['quant_range'] + params['quant_min']
    
    # Denormalize to original scale
    reconstructed = normalizer.denormalize(dequantized, params)
    
    return reconstructed


def compute_quantization_error(original, quantized, bins=1024):
    """
    Compute theoretical quantization error.
    
    Args:
        original (np.ndarray): Original normalized vertices
        quantized (np.ndarray): Quantized vertices
        bins (int): Number of bins
        
    Returns:
        dict: Error metrics
    """
    dequantized = dequantize_vertices(quantized, bins)
    
    error = np.abs(original - dequantized)
    
    return {
        'mean_abs_error': error.mean(),
        'max_abs_error': error.max(),
        'std_error': error.std(),
        'quantization_step': 1.0 / (bins - 1)
    }


if __name__ == "__main__":
    # Test quantization
    print("Testing Quantization Methods\n")
    
    # Create sample normalized vertices [0, 1]
    vertices = np.array([
        [0.0, 0.5, 1.0],
        [0.25, 0.75, 0.3],
        [0.1, 0.9, 0.6]
    ])
    
    print("Original normalized vertices:")
    print(vertices)
    
    bins = 1024
    print(f"\nQuantizing with {bins} bins...")
    
    # Quantize
    quantized = quantize_vertices(vertices, bins)
    print("Quantized (integer bins):")
    print(quantized)
    
    # Dequantize
    dequantized = dequantize_vertices(quantized, bins)
    print("\nDequantized:")
    print(dequantized)
    
    # Compute error
    error = np.abs(vertices - dequantized)
    print("\nAbsolute Error:")
    print(error)
    print(f"\nMax error: {error.max():.6f}")
    print(f"Mean error: {error.mean():.6f}")
    print(f"Theoretical max error: {1.0 / (bins - 1):.6f}")
