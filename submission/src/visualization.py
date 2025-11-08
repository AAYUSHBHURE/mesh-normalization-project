"""
Visualization utilities for 3D meshes.
Uses Open3D for 3D visualization.
"""

import numpy as np
import os

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. 3D visualization will be disabled.")


def create_mesh_o3d(vertices, faces=None):
    """
    Create an Open3D mesh object.
    
    Args:
        vertices (np.ndarray): Nx3 array of vertex coordinates
        faces (np.ndarray): Mx3 array of face indices (optional)
        
    Returns:
        o3d.geometry.TriangleMesh or o3d.geometry.PointCloud
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Cannot create mesh.")
        return None
    
    if faces is not None and len(faces) > 0:
        # Create triangle mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        return mesh
    else:
        # Create point cloud if no faces
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        return pcd


def visualize_mesh(vertices, faces=None, window_name="3D Mesh", point_size=2.0):
    """
    Visualize a mesh using Open3D.
    
    Args:
        vertices (np.ndarray): Nx3 array of vertex coordinates
        faces (np.ndarray): Mx3 array of face indices (optional)
        window_name (str): Window title
        point_size (float): Point size for point cloud visualization
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Skipping visualization.")
        return
    
    mesh_obj = create_mesh_o3d(vertices, faces)
    
    if mesh_obj is None:
        return
    
    # Set color
    if isinstance(mesh_obj, o3d.geometry.TriangleMesh):
        mesh_obj.paint_uniform_color([0.7, 0.7, 0.7])
    else:
        # Point cloud
        colors = np.tile([0.7, 0.7, 0.7], (len(vertices), 1))
        mesh_obj.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(mesh_obj)
    
    # Set point size if it's a point cloud
    if isinstance(mesh_obj, o3d.geometry.PointCloud):
        render_option = vis.get_render_option()
        render_option.point_size = point_size
    
    vis.run()
    vis.destroy_window()


def visualize_comparison(original_vertices, reconstructed_vertices, faces=None,
                        original_name="Original", reconstructed_name="Reconstructed"):
    """
    Visualize original and reconstructed meshes side by side.
    
    Args:
        original_vertices (np.ndarray): Original vertices
        reconstructed_vertices (np.ndarray): Reconstructed vertices
        faces (np.ndarray): Face indices (optional)
        original_name (str): Label for original mesh
        reconstructed_name (str): Label for reconstructed mesh
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Skipping comparison visualization.")
        return
    
    # Create meshes
    mesh_orig = create_mesh_o3d(original_vertices, faces)
    mesh_recon = create_mesh_o3d(reconstructed_vertices, faces)
    
    if mesh_orig is None or mesh_recon is None:
        return
    
    # Offset reconstructed mesh for side-by-side view
    offset = np.array([original_vertices.max(axis=0)[0] - original_vertices.min(axis=0)[0] + 1, 0, 0])
    if isinstance(mesh_recon, o3d.geometry.TriangleMesh):
        mesh_recon.vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh_recon.vertices) + offset
        )
    else:
        mesh_recon.points = o3d.utility.Vector3dVector(
            np.asarray(mesh_recon.points) + offset
        )
    
    # Set colors
    if isinstance(mesh_orig, o3d.geometry.TriangleMesh):
        mesh_orig.paint_uniform_color([0.3, 0.6, 0.9])  # Blue
        mesh_recon.paint_uniform_color([0.9, 0.3, 0.3])  # Red
    else:
        orig_colors = np.tile([0.3, 0.6, 0.9], (len(original_vertices), 1))
        recon_colors = np.tile([0.9, 0.3, 0.3], (len(reconstructed_vertices), 1))
        mesh_orig.colors = o3d.utility.Vector3dVector(orig_colors)
        mesh_recon.colors = o3d.utility.Vector3dVector(recon_colors)
    
    # Visualize
    o3d.visualization.draw_geometries(
        [mesh_orig, mesh_recon],
        window_name=f"{original_name} (Blue) vs {reconstructed_name} (Red)"
    )


def save_mesh_screenshot(vertices, faces=None, output_path='output/visualizations/mesh.png'):
    """
    Save a screenshot of the mesh.
    
    Args:
        vertices (np.ndarray): Vertex coordinates
        faces (np.ndarray): Face indices (optional)
        output_path (str): Path to save the screenshot
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Cannot save screenshot.")
        return
    
    mesh_obj = create_mesh_o3d(vertices, faces)
    
    if mesh_obj is None:
        return
    
    # Set color
    if isinstance(mesh_obj, o3d.geometry.TriangleMesh):
        mesh_obj.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Create visualizer in off-screen mode
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh_obj)
    vis.update_geometry(mesh_obj)
    vis.poll_events()
    vis.update_renderer()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Capture and save
    vis.capture_screen_image(output_path)
    vis.destroy_window()
    
    print(f"✓ Saved mesh screenshot: {output_path}")


def save_mesh_file(vertices, faces, output_path, file_format='ply'):
    """
    Save mesh to file.
    
    Args:
        vertices (np.ndarray): Vertex coordinates
        faces (np.ndarray): Face indices
        output_path (str): Path to save the mesh
        file_format (str): File format ('ply' or 'obj')
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Using fallback mesh saving.")
        # Fallback: save as simple OBJ format
        save_obj_simple(vertices, faces, output_path)
        return
    
    mesh = create_mesh_o3d(vertices, faces)
    
    if mesh is None:
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save mesh
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"✓ Saved mesh: {output_path}")


def save_obj_simple(vertices, faces, output_path):
    """
    Simple OBJ file writer (fallback when Open3D not available).
    
    Args:
        vertices (np.ndarray): Vertex coordinates
        faces (np.ndarray): Face indices
        output_path (str): Path to save the OBJ file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces (OBJ indices are 1-based)
        if faces is not None:
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"✓ Saved mesh (simple OBJ): {output_path}")


if __name__ == "__main__":
    print("Testing Visualization Module\n")
    
    if OPEN3D_AVAILABLE:
        # Create a simple cube for testing
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
        ], dtype=np.float64)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ], dtype=np.int32)
        
        print("Visualizing test cube...")
        visualize_mesh(vertices, faces, "Test Cube")
    else:
        print("Open3D not available. Install with: pip install open3d")
