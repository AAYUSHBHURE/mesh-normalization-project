"""
Visualize detected UV seams on the mesh.
Creates plots showing the 3D mesh with seam edges highlighted.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from seam_tokenization import SeamTokenizer, create_example_cube_with_uvs
import os


def visualize_mesh_with_seams(vertices, faces, seams, output_path):
    """
    Visualize a 3D mesh with seam edges highlighted in red.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices
        seams: List of SeamEdge objects
        output_path: Path to save the visualization
    """
    fig = plt.figure(figsize=(15, 5))
    
    # View 1: Front view with seams
    ax1 = fig.add_subplot(131, projection='3d')
    plot_mesh_with_seams(ax1, vertices, faces, seams, title="Front View")
    ax1.view_init(elev=20, azim=45)
    
    # View 2: Top view with seams
    ax2 = fig.add_subplot(132, projection='3d')
    plot_mesh_with_seams(ax2, vertices, faces, seams, title="Top View")
    ax2.view_init(elev=90, azim=0)
    
    # View 3: Side view with seams
    ax3 = fig.add_subplot(133, projection='3d')
    plot_mesh_with_seams(ax3, vertices, faces, seams, title="Side View")
    ax3.view_init(elev=0, azim=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved seam visualization: {output_path}")
    plt.close()


def plot_mesh_with_seams(ax, vertices, faces, seams, title="Mesh with Seams"):
    """Plot mesh wireframe with seam edges highlighted."""
    # Draw all edges in light gray
    edges = set()
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            edges.add(edge)
    
    # Separate seam edges from regular edges
    seam_edge_set = set()
    for seam in seams:
        seam_edge_set.add(tuple(sorted(seam.vertex_ids)))
    
    regular_edges = edges - seam_edge_set
    
    # Draw regular edges (light gray)
    regular_edge_segments = []
    for v1, v2 in regular_edges:
        regular_edge_segments.append([vertices[v1], vertices[v2]])
    
    if regular_edge_segments:
        regular_lines = Line3DCollection(regular_edge_segments, colors='lightgray', 
                                        linewidths=0.5, alpha=0.6)
        ax.add_collection3d(regular_lines)
    
    # Draw seam edges (thick red)
    seam_edge_segments = []
    for seam in seams:
        v1, v2 = seam.vertex_ids
        seam_edge_segments.append([vertices[v1], vertices[v2]])
    
    if seam_edge_segments:
        seam_lines = Line3DCollection(seam_edge_segments, colors='red', 
                                     linewidths=3, alpha=1.0, label='Seams')
        ax.add_collection3d(seam_lines)
    
    # Draw vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              c='blue', s=30, alpha=0.6, depthshade=True)
    
    # Highlight seam vertices in red
    seam_vertex_ids = set()
    for seam in seams:
        seam_vertex_ids.update(seam.vertex_ids)
    
    if seam_vertex_ids:
        seam_verts = np.array([vertices[i] for i in seam_vertex_ids])
        ax.scatter(seam_verts[:, 0], seam_verts[:, 1], seam_verts[:, 2],
                  c='red', s=80, alpha=1.0, depthshade=True, 
                  edgecolors='darkred', linewidths=2, label='Seam Vertices')
    
    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontweight='bold')
    
    # Equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                         vertices[:, 1].max() - vertices[:, 1].min(),
                         vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def visualize_uv_map_with_seams(uvs, face_uvs, seams, output_path):
    """
    Visualize the UV map with seam edges highlighted.
    
    Args:
        uvs: (M, 2) UV coordinates
        face_uvs: (F, 3) UV indices per face
        seams: List of SeamEdge objects
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw UV triangles
    for face_uv in face_uvs:
        triangle = np.array([uvs[face_uv[0]], uvs[face_uv[1]], uvs[face_uv[2]], uvs[face_uv[0]]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    
    # Draw all UV points
    ax.scatter(uvs[:, 0], uvs[:, 1], c='blue', s=30, alpha=0.6, zorder=3)
    
    # Highlight seam UVs
    for seam in seams:
        # Draw both UV coordinates for this seam
        uv1 = np.array(seam.uv_coords_1)
        uv2 = np.array(seam.uv_coords_2)
        
        # Plot the two different UV positions for the same edge
        ax.scatter(*uv1, c='red', s=150, marker='o', edgecolors='darkred', 
                  linewidths=2, zorder=5, alpha=0.8)
        ax.scatter(*uv2, c='orange', s=150, marker='s', edgecolors='darkorange',
                  linewidths=2, zorder=5, alpha=0.8)
        
        # Draw line connecting the two UV representations
        ax.plot([uv1[0], uv2[0]], [uv1[1], uv2[1]], 'r--', linewidth=2, 
               alpha=0.7, zorder=4)
        
        # Add labels
        ax.text(uv1[0], uv1[1], f'  UV1\n  {seam.vertex_ids}', 
               fontsize=8, ha='left', va='bottom', color='darkred', fontweight='bold')
        ax.text(uv2[0], uv2[1], f'  UV2\n  {seam.vertex_ids}', 
               fontsize=8, ha='left', va='top', color='darkorange', fontweight='bold')
    
    ax.set_xlabel('U Coordinate', fontsize=12, fontweight='bold')
    ax.set_ylabel('V Coordinate', fontsize=12, fontweight='bold')
    ax.set_title('UV Map with Seam Discontinuities\n' + 
                'Red circles = UV1, Orange squares = UV2 (same 3D edge, different UV)', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='darkred', label='Seam UV Position 1'),
        Patch(facecolor='orange', edgecolor='darkorange', label='Seam UV Position 2'),
        plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='UV Discontinuity')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved UV map visualization: {output_path}")
    plt.close()


def create_seam_info_plot(seams, output_path):
    """Create an informative plot showing seam statistics."""
    if not seams:
        print("No seams to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Seam types distribution
    ax1 = axes[0, 0]
    seam_types = [seam.seam_type.name for seam in seams]
    unique_types, counts = np.unique(seam_types, return_counts=True)
    ax1.bar(unique_types, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax1.set_xlabel('Seam Type', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Seam Type Distribution', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: UV discontinuity magnitudes
    ax2 = axes[0, 1]
    u_diffs = []
    v_diffs = []
    for seam in seams:
        u_diff = abs(seam.uv_coords_1[0] - seam.uv_coords_2[0])
        v_diff = abs(seam.uv_coords_1[1] - seam.uv_coords_2[1])
        u_diffs.append(u_diff)
        v_diffs.append(v_diff)
    
    x = np.arange(len(seams))
    width = 0.35
    ax2.bar(x - width/2, u_diffs, width, label='U Discontinuity', color='#FF6B6B', alpha=0.8)
    ax2.bar(x + width/2, v_diffs, width, label='V Discontinuity', color='#4ECDC4', alpha=0.8)
    ax2.set_xlabel('Seam Index', fontweight='bold')
    ax2.set_ylabel('UV Discontinuity Magnitude', fontweight='bold')
    ax2.set_title('UV Coordinate Discontinuities per Seam', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Seam details table
    ax3 = axes[1, 0]
    ax3.axis('tight')
    ax3.axis('off')
    
    table_data = []
    headers = ['Seam #', 'Vertices', 'UV1 (u,v)', 'UV2 (u,v)', 'Type']
    
    for i, seam in enumerate(seams):
        row = [
            str(i+1),
            f"({seam.vertex_ids[0]}, {seam.vertex_ids[1]})",
            f"({seam.uv_coords_1[0]:.2f}, {seam.uv_coords_1[1]:.2f})",
            f"({seam.uv_coords_2[0]:.2f}, {seam.uv_coords_2[1]:.2f})",
            seam.seam_type.name
        ]
        table_data.append(row)
    
    table = ax3.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.1, 0.2, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Seam Details', fontweight='bold', fontsize=12, pad=20)
    
    # Plot 4: Token encoding visualization
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Show token encoding for first seam
    tokenizer = SeamTokenizer(quantization_bits=8)
    if seams:
        tokens = tokenizer.encode_seam(seams[0])
        
        info_text = f"""
Token Encoding Example (Seam #1):

Raw Token Sequence:
{tokens}

Token Breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Position 0: SEAM_START = {tokens[0]}
  Position 1-2: Vertex IDs = ({tokens[1]}, {tokens[2]})
  Position 3-4: UV1 quantized = ({tokens[3]}, {tokens[4]})
  Position 5-6: UV2 quantized = ({tokens[5]}, {tokens[6]})
  Position 7: Seam Type = {tokens[7]} ({seams[0].seam_type.name})
  Position 8: SEAM_END = {tokens[8]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Tokens: {len(tokens)}
Compression: 9 tokens per seam
Quantization: 8-bit (256 levels)

This compact representation enables:
• Transformer model processing
• Efficient mesh encoding
• SeamGPT-style generation
        """
        
        ax4.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
    
    plt.suptitle('UV Seam Analysis and Tokenization', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved seam analysis: {output_path}")
    plt.close()


def main():
    """Generate all seam visualizations."""
    print("="*70)
    print("GENERATING SEAM VISUALIZATIONS")
    print("="*70)
    
    # Create output directory
    output_dir = "output/seam_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create example mesh with UVs
    print("\n1. Loading mesh with UV mapping...")
    vertices, faces, uvs, face_uvs = create_example_cube_with_uvs()
    
    # Detect seams
    print("2. Detecting UV seams...")
    tokenizer = SeamTokenizer(quantization_bits=8)
    seams = tokenizer.detect_seams(vertices, faces, uvs, face_uvs)
    print(f"   Found {len(seams)} seam(s)")
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    
    # 3D mesh with seams highlighted
    visualize_mesh_with_seams(
        vertices, faces, seams,
        os.path.join(output_dir, "mesh_with_seams_3d.png")
    )
    
    # UV map with seams
    visualize_uv_map_with_seams(
        uvs, face_uvs, seams,
        os.path.join(output_dir, "uv_map_with_seams.png")
    )
    
    # Seam analysis and info
    create_seam_info_plot(
        seams,
        os.path.join(output_dir, "seam_analysis.png")
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files in: {output_dir}/")
    print("  - mesh_with_seams_3d.png: 3D mesh with seam edges in red")
    print("  - uv_map_with_seams.png: UV layout showing discontinuities")
    print("  - seam_analysis.png: Statistics and token encoding details")
    print("="*70)


if __name__ == "__main__":
    main()
