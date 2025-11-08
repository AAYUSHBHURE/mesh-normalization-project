"""
Bonus Task: Seam Tokenization Prototype
Goal: Represent mesh seams as discrete tokens for SeamGPT-style processing.

Seams are edges where UV mapping breaks - vertices that share the same 3D position
but have different UV coordinates, creating discontinuities in texture mapping.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    """Token types for seam encoding."""
    SEAM_START = 0
    SEAM_END = 1
    VERTEX_3D = 2
    UV_COORD = 3
    EDGE = 4
    SEAM_TYPE = 5


class SeamType(Enum):
    """Types of UV seams."""
    HORIZONTAL = 0  # U wraps around (0 -> 1)
    VERTICAL = 1    # V wraps around (0 -> 1)
    DIAGONAL = 2    # Both U and V discontinuity
    BOUNDARY = 3    # Edge of UV map


@dataclass
class SeamEdge:
    """Represents a seam edge in the mesh."""
    vertex_ids: Tuple[int, int]  # 3D vertex indices
    uv_coords_1: Tuple[float, float]  # UV at first occurrence
    uv_coords_2: Tuple[float, float]  # UV at second occurrence
    seam_type: SeamType
    
    def __repr__(self):
        return f"Seam({self.vertex_ids}, UV1={self.uv_coords_1}, UV2={self.uv_coords_2}, type={self.seam_type.name})"


class SeamTokenizer:
    """
    Encodes and decodes mesh seams as token sequences.
    
    Token Format:
    [SEAM_START, v1_id, v2_id, u1, v1, u2, v2, seam_type, SEAM_END]
    """
    
    def __init__(self, quantization_bits: int = 8):
        """
        Initialize tokenizer.
        
        Args:
            quantization_bits: Bits for quantizing UV coordinates (default: 8 bits = 256 levels)
        """
        self.quantization_bits = quantization_bits
        self.max_quant_value = (1 << quantization_bits) - 1
    
    def detect_seams(self, vertices: np.ndarray, faces: np.ndarray, 
                     uvs: np.ndarray, face_uvs: np.ndarray) -> List[SeamEdge]:
        """
        Detect UV seams in a mesh.
        
        Args:
            vertices: (N, 3) vertex positions
            faces: (F, 3) face vertex indices
            uvs: (M, 2) UV coordinates
            face_uvs: (F, 3) UV indices per face
            
        Returns:
            List of SeamEdge objects
        """
        seams = []
        edge_uv_map = {}  # Map (v1, v2) -> list of UV pairs
        
        # Build edge-UV mapping
        for face_idx, (face, face_uv) in enumerate(zip(faces, face_uvs)):
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                uv1, uv2 = uvs[face_uv[i]], uvs[face_uv[(i + 1) % 3]]
                
                # Normalize edge direction (smaller index first)
                edge = tuple(sorted([v1, v2]))
                uv_pair = (tuple(uv1), tuple(uv2)) if v1 < v2 else (tuple(uv2), tuple(uv1))
                
                if edge not in edge_uv_map:
                    edge_uv_map[edge] = []
                edge_uv_map[edge].append(uv_pair)
        
        # Find seams: edges with different UV coordinates
        for edge, uv_pairs in edge_uv_map.items():
            if len(uv_pairs) > 1:
                # Check if UV coordinates differ
                uv1_set = set(uv_pairs[0])
                for uv_pair in uv_pairs[1:]:
                    uv2_set = set(uv_pair)
                    if uv1_set != uv2_set:
                        # Found a seam
                        seam_type = self._classify_seam(uv_pairs[0], uv_pair)
                        seam = SeamEdge(
                            vertex_ids=edge,
                            uv_coords_1=uv_pairs[0][0],
                            uv_coords_2=uv_pair[0],
                            seam_type=seam_type
                        )
                        seams.append(seam)
        
        return seams
    
    def _classify_seam(self, uv1: Tuple, uv2: Tuple, threshold: float = 0.9) -> SeamType:
        """Classify the type of UV seam based on UV discontinuity."""
        u1_avg = (uv1[0][0] + uv1[1][0]) / 2
        v1_avg = (uv1[0][1] + uv1[1][1]) / 2
        u2_avg = (uv2[0][0] + uv2[1][0]) / 2
        v2_avg = (uv2[0][1] + uv2[1][1]) / 2
        
        u_diff = abs(u1_avg - u2_avg)
        v_diff = abs(v1_avg - v2_avg)
        
        if u_diff > threshold and v_diff > threshold:
            return SeamType.DIAGONAL
        elif u_diff > threshold:
            return SeamType.HORIZONTAL
        elif v_diff > threshold:
            return SeamType.VERTICAL
        else:
            return SeamType.BOUNDARY
    
    def encode_seam(self, seam: SeamEdge) -> List[int]:
        """
        Encode a single seam into a token sequence.
        
        Token format:
        [SEAM_START, v1_id, v2_id, u1_quant, v1_quant, u2_quant, v2_quant, seam_type, SEAM_END]
        """
        tokens = [TokenType.SEAM_START.value]
        
        # Add vertex IDs
        tokens.extend([seam.vertex_ids[0], seam.vertex_ids[1]])
        
        # Quantize and add UV coordinates
        u1_q = int(np.clip(seam.uv_coords_1[0] * self.max_quant_value, 0, self.max_quant_value))
        v1_q = int(np.clip(seam.uv_coords_1[1] * self.max_quant_value, 0, self.max_quant_value))
        u2_q = int(np.clip(seam.uv_coords_2[0] * self.max_quant_value, 0, self.max_quant_value))
        v2_q = int(np.clip(seam.uv_coords_2[1] * self.max_quant_value, 0, self.max_quant_value))
        
        tokens.extend([u1_q, v1_q, u2_q, v2_q])
        
        # Add seam type
        tokens.append(seam.seam_type.value)
        
        # End token
        tokens.append(TokenType.SEAM_END.value)
        
        return tokens
    
    def decode_seam(self, tokens: List[int]) -> Optional[SeamEdge]:
        """
        Decode a token sequence back into a SeamEdge.
        
        Args:
            tokens: Token sequence
            
        Returns:
            SeamEdge or None if invalid
        """
        if len(tokens) < 9:
            return None
        
        if tokens[0] != TokenType.SEAM_START.value or tokens[-1] != TokenType.SEAM_END.value:
            return None
        
        v1_id = tokens[1]
        v2_id = tokens[2]
        u1_q, v1_q = tokens[3], tokens[4]
        u2_q, v2_q = tokens[5], tokens[6]
        seam_type_val = tokens[7]
        
        # Dequantize UV coordinates
        u1 = u1_q / self.max_quant_value
        v1 = v1_q / self.max_quant_value
        u2 = u2_q / self.max_quant_value
        v2 = v2_q / self.max_quant_value
        
        return SeamEdge(
            vertex_ids=(v1_id, v2_id),
            uv_coords_1=(u1, v1),
            uv_coords_2=(u2, v2),
            seam_type=SeamType(seam_type_val)
        )
    
    def encode_mesh_seams(self, seams: List[SeamEdge]) -> List[int]:
        """Encode all seams in a mesh as a single token sequence."""
        all_tokens = []
        for seam in seams:
            all_tokens.extend(self.encode_seam(seam))
        return all_tokens
    
    def decode_mesh_seams(self, tokens: List[int]) -> List[SeamEdge]:
        """Decode a token sequence into multiple seams."""
        seams = []
        i = 0
        while i < len(tokens):
            if tokens[i] == TokenType.SEAM_START.value:
                # Find matching end token
                end_idx = i + 1
                while end_idx < len(tokens) and tokens[end_idx] != TokenType.SEAM_END.value:
                    end_idx += 1
                
                if end_idx < len(tokens):
                    seam_tokens = tokens[i:end_idx + 1]
                    seam = self.decode_seam(seam_tokens)
                    if seam:
                        seams.append(seam)
                    i = end_idx + 1
                else:
                    break
            else:
                i += 1
        
        return seams


def create_example_cube_with_uvs():
    """
    Create a simple cube with UV mapping that has seams.
    
    A cube unwrapped has seams along edges where the UV map wraps.
    """
    # Cube vertices (8 corners)
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Front face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Back face
    ])
    
    # Faces (12 triangles = 6 quad faces split)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Front
        [4, 7, 6], [4, 6, 5],  # Back
        [0, 4, 5], [0, 5, 1],  # Bottom
        [2, 6, 7], [2, 7, 3],  # Top
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 5, 6], [1, 6, 2]   # Right
    ])
    
    # UV coordinates (unwrapped cube - creates seams)
    # Simple cross pattern unwrapping
    uvs = np.array([
        # Front face
        [0.25, 0.33], [0.5, 0.33], [0.5, 0.66], [0.25, 0.66],
        # Back face (wraps around)
        [0.75, 0.33], [1.0, 0.33], [1.0, 0.66], [0.75, 0.66],
        # Additional UVs for seams
        [0.0, 0.33], [0.25, 0.0], [0.5, 0.0], [0.5, 1.0], [0.25, 1.0]
    ])
    
    # UV indices per face
    face_uvs = np.array([
        [0, 1, 2], [0, 2, 3],   # Front
        [4, 7, 6], [4, 6, 5],   # Back
        [8, 4, 5], [8, 5, 1],   # Bottom (seam at edge 0-1)
        [2, 6, 7], [2, 7, 3],   # Top
        [0, 3, 7], [0, 7, 4],   # Left
        [1, 5, 6], [1, 6, 2]    # Right
    ])
    
    return vertices, faces, uvs, face_uvs


def demonstrate_seam_tokenization():
    """Demonstrate the seam tokenization system."""
    print("="*70)
    print("SEAM TOKENIZATION PROTOTYPE - Bonus Task")
    print("="*70)
    print("\nGoal: Represent mesh UV seams as discrete tokens for SeamGPT-style processing\n")
    
    # Create example mesh
    print("1. Creating example cube mesh with UV mapping...")
    vertices, faces, uvs, face_uvs = create_example_cube_with_uvs()
    print(f"   - Vertices: {len(vertices)}")
    print(f"   - Faces: {len(faces)}")
    print(f"   - UV coordinates: {len(uvs)}")
    
    # Initialize tokenizer
    print("\n2. Initializing seam tokenizer (8-bit quantization)...")
    tokenizer = SeamTokenizer(quantization_bits=8)
    
    # Detect seams
    print("\n3. Detecting UV seams...")
    seams = tokenizer.detect_seams(vertices, faces, uvs, face_uvs)
    print(f"   Found {len(seams)} seam(s)")
    
    if len(seams) > 0:
        print("\n4. Example seam details:")
        for i, seam in enumerate(seams[:3]):  # Show first 3
            print(f"   Seam {i+1}: {seam}")
        
        # Encode first seam
        print("\n5. Encoding first seam as tokens...")
        tokens = tokenizer.encode_seam(seams[0])
        print(f"   Token sequence: {tokens}")
        print(f"   Token breakdown:")
        print(f"     - SEAM_START: {tokens[0]}")
        print(f"     - Vertex IDs: {tokens[1]}, {tokens[2]}")
        print(f"     - UV1 quantized: ({tokens[3]}, {tokens[4]})")
        print(f"     - UV2 quantized: ({tokens[5]}, {tokens[6]})")
        print(f"     - Seam type: {SeamType(tokens[7]).name}")
        print(f"     - SEAM_END: {tokens[8]}")
        
        # Decode
        print("\n6. Decoding tokens back to seam...")
        decoded_seam = tokenizer.decode_seam(tokens)
        print(f"   Reconstructed: {decoded_seam}")
        
        # Encode all seams
        print("\n7. Encoding all mesh seams...")
        all_tokens = tokenizer.encode_mesh_seams(seams)
        print(f"   Total tokens: {len(all_tokens)}")
        print(f"   Token sequence (first 30): {all_tokens[:30]}...")
        
        # Decode all
        print("\n8. Decoding all seams...")
        decoded_seams = tokenizer.decode_mesh_seams(all_tokens)
        print(f"   Reconstructed {len(decoded_seams)} seam(s)")
        
        # Verify
        print("\n9. Verification:")
        match = len(seams) == len(decoded_seams)
        print(f"   Encoding/decoding successful: {match}")
    
    print("\n" + "="*70)
    print("CONNECTION TO SEAMGPT AND MESH UNDERSTANDING")
    print("="*70)
    print("""
This tokenization scheme enables:
1. Sequential processing: Seams represented as token sequences can be 
   processed by transformer models like GPT for mesh understanding.
   
2. Compression: UV discontinuities encoded compactly (9 tokens per seam)
   enables efficient mesh representation for AI training.
   
3. Generative modeling: Token-based representation allows SeamGPT-style
   models to generate or modify UV layouts by predicting token sequences.
   
4. Semantic understanding: Seam types (horizontal, vertical, diagonal)
   provide structural information about mesh topology and UV unwrapping
   strategies, helping AI understand 3D geometry organization.
   
5. Mesh repair: By learning seam patterns, models can detect and fix
   UV mapping errors or suggest optimal unwrapping strategies.

This approach bridges 3D geometry with sequence modeling, enabling
transformer architectures to process 3D meshes as they do text/code.
    """)
    print("="*70)


if __name__ == "__main__":
    demonstrate_seam_tokenization()
