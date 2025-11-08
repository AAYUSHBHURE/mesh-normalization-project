"""Generate mesh state visualizations before normalization, after normalization,
and after quantization/reconstruction for each sample mesh and normalizer.
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend for headless execution
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

from src.mesh_loader import load_mesh
from src.normalization import MinMaxNormalizer, UnitSphereNormalizer
from src.quantization import quantize_mesh, dequantize_mesh

# Directory where visualizations will be written
OUTPUT_DIR = Path("output/visualizations")

# Mesh files to visualize (relative to project root)
MESH_NAMES = ["cube", "pyramid", "sphere", "tetrahedron"]
MESH_DIR = Path("data")

# Normalizers to inspect
NORMALIZERS = {
    "MinMax": MinMaxNormalizer(),
    "UnitSphere": UnitSphereNormalizer(),
}

BINS = 1024

def plot_mesh(mesh: trimesh.Trimesh, title: str, output_path: Path) -> None:
    """Render a mesh with Matplotlib and save to ``output_path``."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Build triangle collection for plotting
    triangles = mesh.vertices[mesh.faces]
    collection = Poly3DCollection(
        triangles,
        alpha=0.85,
        facecolor="#66c2a5",
        edgecolor="#2c7fb8",
        linewidths=0.2,
    )
    ax.add_collection3d(collection)

    # Create symmetric viewing window around mesh centroid
    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    max_radius = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2 or 1.0

    ax.set_xlim(centroid[0] - max_radius, centroid[0] + max_radius)
    ax.set_ylim(centroid[1] - max_radius, centroid[1] + max_radius)
    ax.set_zlim(centroid[2] - max_radius, centroid[2] + max_radius)
    ax.set_box_aspect([1.0, 1.0, 1.0])

    ax.view_init(elev=20, azim=35)
    ax.set_title(title)
    ax.set_axis_off()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def build_trimesh_from_vertices(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Construct a mesh using ``vertices`` and ``faces`` arrays."""
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for mesh_name in MESH_NAMES:
        mesh_path = MESH_DIR / f"{mesh_name}.obj"
        vertices, faces = load_mesh(mesh_path)
        mesh = build_trimesh_from_vertices(vertices, faces)

        base_output = OUTPUT_DIR / mesh_name
        base_output.mkdir(parents=True, exist_ok=True)

        # 1) Original mesh
        plot_mesh(mesh, f"{mesh_name.title()} - Original", base_output / f"{mesh_name}_original.png")

        # 2) Process each normalizer pipeline
        for method_name, normalizer in NORMALIZERS.items():
            normalized_vertices, params = normalizer.normalize(vertices)
            normalized_mesh = build_trimesh_from_vertices(normalized_vertices, faces)
            plot_mesh(
                normalized_mesh,
                f"{mesh_name.title()} - {method_name} Normalized",
                base_output / f"{mesh_name}_{method_name}_normalized.png",
            )

            quantized_vertices, quant_params = quantize_mesh(vertices, normalizer, bins=BINS)
            reconstructed_vertices = dequantize_mesh(quantized_vertices, quant_params, normalizer)
            reconstructed_mesh = build_trimesh_from_vertices(reconstructed_vertices, faces)
            plot_mesh(
                reconstructed_mesh,
                f"{mesh_name.title()} - {method_name} Reconstructed",
                base_output / f"{mesh_name}_{method_name}_reconstructed.png",
            )

if __name__ == "__main__":
    main()
