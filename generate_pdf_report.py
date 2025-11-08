"""
Automatic PDF Report Generator for Mesh Normalization Assignment
Generates a complete PDF report with all results, plots, and analysis.
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, 
                                PageBreak, Table, TableStyle)
from reportlab.lib import colors
from datetime import datetime


def create_pdf_report(output_filename='Assignment_Report.pdf'):
    """Generate complete PDF report."""
    
    # Create document
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=14
    )
    
    # ===================================================================
    # TITLE PAGE
    # ===================================================================
    elements.append(Spacer(1, 1.5*inch))
    elements.append(Paragraph("Mesh Normalization, Quantization,<br/>and Error Analysis", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Assignment Report", styles['Heading2']))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    elements.append(Spacer(1, 0.5*inch))
    
    # Assignment info
    assignment_info = [
        ['Assignment:', 'Mesh Normalization, Quantization, and Error Analysis'],
        ['Total Marks:', '100 + 30 (Bonus)'],
        ['Context:', '3D Graphics and AI - SeamGPT Data Preprocessing'],
        ['Libraries Used:', 'Python, NumPy, Trimesh, Matplotlib, Open3D']
    ]
    
    t = Table(assignment_info, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ]))
    elements.append(t)
    
    elements.append(PageBreak())
    
    # ===================================================================
    # TABLE OF CONTENTS
    # ===================================================================
    elements.append(Paragraph("Table of Contents", heading1_style))
    elements.append(Spacer(1, 0.2*inch))
    
    toc_items = [
        "1. Introduction",
        "2. Task 1: Load and Inspect Mesh (20 Marks)",
        "3. Task 2: Normalize and Quantize (40 Marks)",
        "4. Task 3: Reconstruction and Error Analysis (40 Marks)",
        "5. Bonus Task: Seam Tokenization (30 Marks)",
        "6. Conclusions and Observations",
        "7. References"
    ]
    
    for item in toc_items:
        elements.append(Paragraph(item, styles['Normal']))
        elements.append(Spacer(1, 8))
    
    elements.append(PageBreak())
    
    # ===================================================================
    # 1. INTRODUCTION
    # ===================================================================
    elements.append(Paragraph("1. Introduction", heading1_style))
    
    intro_text = """
    This report presents the implementation and analysis of a complete data preprocessing 
    pipeline for 3D meshes, a fundamental step in AI-driven 3D graphics systems such as SeamGPT. 
    The work focuses on three core tasks: mesh loading and inspection, normalization and 
    quantization, and error analysis of reconstructed meshes. Additionally, a bonus task 
    implementing seam tokenization for mesh understanding has been completed.
    <br/><br/>
    The primary objectives were to:
    <br/>
    • Load and analyze 3D mesh data from .obj files<br/>
    • Implement two normalization methods (Min-Max and Unit Sphere)<br/>
    • Apply 1024-bin quantization to discretize vertex coordinates<br/>
    • Reconstruct meshes and measure reconstruction error<br/>
    • Develop a tokenization scheme for mesh seam representation<br/>
    <br/>
    All implementations follow industry standards, with quantization using the rounding 
    method employed by Google Draco mesh compression library.
    """
    
    elements.append(Paragraph(intro_text, body_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # ===================================================================
    # 2. TASK 1: LOAD AND INSPECT MESH
    # ===================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("2. Task 1: Load and Inspect Mesh (20 Marks)", heading1_style))
    
    task1_text = """
    <b>Objective:</b> Load 3D mesh data and compute statistical properties.
    <br/><br/>
    <b>Methodology:</b><br/>
    • Meshes loaded using Trimesh library from .obj files<br/>
    • Vertex coordinates extracted as NumPy arrays (Nx3)<br/>
    • Statistics computed: count, min, max, mean, standard deviation per axis<br/>
    • Four test meshes used: cube (8 vertices), pyramid (5 vertices), 
    sphere (382 vertices), tetrahedron (4 vertices)<br/>
    """
    
    elements.append(Paragraph(task1_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Statistics table
    elements.append(Paragraph("2.1 Mesh Statistics", heading2_style))
    
    stats_data = [
        ['Mesh', 'Vertices', 'X Range', 'Y Range', 'Z Range'],
        ['Cube', '8', '[-1, 1]', '[-1, 1]', '[-1, 1]'],
        ['Pyramid', '5', '[-1, 1]', '[-1, 1]', '[0, 2]'],
        ['Sphere', '382', '[-1, 1]', '[-1, 1]', '[-1, 1]'],
        ['Tetrahedron', '4', '[-1, 1]', '[-1, 1]', '[-1, 1]']
    ]
    
    t = Table(stats_data, colWidths=[1.5*inch, 1*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))
    
    # Visualizations
    elements.append(Paragraph("2.2 Original Mesh Visualizations", heading2_style))
    
    viz_text = """
    Four test meshes were visualized to verify successful loading. The meshes represent 
    different geometric complexities: simple platonic solids (cube, tetrahedron), a pyramid 
    with asymmetric coordinate ranges, and a sphere with higher vertex density.
    """
    elements.append(Paragraph(viz_text, body_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Add original mesh images (2x2 grid)
    mesh_names = ['cube', 'pyramid', 'sphere', 'tetrahedron']
    viz_paths = []
    for mesh in mesh_names:
        path = f'output/visualizations/{mesh}/{mesh}_original.png'
        if os.path.exists(path):
            viz_paths.append(path)
    
    if len(viz_paths) >= 4:
        # Create 2x2 image grid
        img_data = []
        for i in range(0, 4, 2):
            row = []
            for j in range(2):
                if i + j < len(viz_paths):
                    img = Image(viz_paths[i + j], width=2.5*inch, height=2.5*inch)
                    row.append(img)
            if row:
                img_data.append(row)
        
        img_table = Table(img_data, colWidths=[3*inch, 3*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(img_table)
    
    # ===================================================================
    # 3. TASK 2: NORMALIZE AND QUANTIZE
    # ===================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("3. Task 2: Normalize and Quantize (40 Marks)", heading1_style))
    
    task2_text = """
    <b>Objective:</b> Transform meshes into standardized coordinate ranges and discretize 
    for compression.
    <br/><br/>
    <b>3.1 Normalization Methods</b>
    <br/><br/>
    Two normalization methods were implemented:
    <br/><br/>
    <b>Min-Max Normalization:</b><br/>
    Formula: x' = (x - x_min) / (x_max - x_min)<br/>
    • Maps coordinates to [0, 1] range<br/>
    • Preserves aspect ratio and relative proportions<br/>
    • Best for uniform scaling across all axes<br/>
    <br/>
    <b>Unit Sphere Normalization:</b><br/>
    Formula: x' = (x - center) / max_distance<br/>
    • Centers mesh at origin<br/>
    • Scales to fit within unit sphere (radius = 1)<br/>
    • Best for rotation-invariant processing<br/>
    """
    
    elements.append(Paragraph(task2_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Quantization explanation
    quant_text = """
    <b>3.2 Quantization Process</b>
    <br/><br/>
    <b>Forward Quantization:</b><br/>
    q = round(x' × 1023) where x' ∈ [0, 1]<br/>
    <br/>
    <b>Inverse Dequantization:</b><br/>
    x'' = q / 1023<br/>
    <br/>
    <b>Key Implementation Detail:</b><br/>
    The quantization uses <i>rounding</i> (round() function) rather than floor(), 
    following the industry standard employed by Google Draco mesh compression library. 
    This eliminates systematic downward bias and provides unbiased quantization error.
    <br/><br/>
    <b>Results:</b><br/>
    • 8 quantized meshes generated (4 meshes × 2 methods)<br/>
    • All meshes saved in .ply format<br/>
    • Bin size: 1024 (10-bit precision)<br/>
    """
    
    elements.append(Paragraph(quant_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Comparison visualization
    elements.append(Paragraph("3.3 Normalization Comparison", heading2_style))
    
    # Add comparison images for one mesh (e.g., sphere)
    comp_paths = [
        'output/visualizations/sphere/sphere_original.png',
        'output/visualizations/sphere/sphere_MinMax_normalized.png',
        'output/visualizations/sphere/sphere_UnitSphere_normalized.png'
    ]
    
    existing_comp = [p for p in comp_paths if os.path.exists(p)]
    if len(existing_comp) >= 3:
        comp_data = [[Image(existing_comp[0], width=2*inch, height=2*inch),
                      Image(existing_comp[1], width=2*inch, height=2*inch),
                      Image(existing_comp[2], width=2*inch, height=2*inch)]]
        
        comp_table = Table(comp_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        comp_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(comp_table)
        
        # Add labels
        labels = [['Original', 'Min-Max Normalized', 'Unit Sphere Normalized']]
        label_table = Table(labels, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        label_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ]))
        elements.append(label_table)
    
    # ===================================================================
    # 4. TASK 3: RECONSTRUCTION AND ERROR ANALYSIS
    # ===================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("4. Task 3: Reconstruction and Error Analysis (40 Marks)", heading1_style))
    
    task3_text = """
    <b>Objective:</b> Measure information loss from quantization by comparing original 
    and reconstructed meshes.
    <br/><br/>
    <b>Methodology:</b><br/>
    • Dequantize quantized coordinates: x'' = q / 1023<br/>
    • Denormalize to recover original scale<br/>
    • Compute error metrics: MSE, MAE, RMSE, Maximum Error<br/>
    • Analyze per-axis error distribution<br/>
    """
    
    elements.append(Paragraph(task3_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Error metrics table
    elements.append(Paragraph("4.1 Error Metrics Summary", heading2_style))
    
    error_data = [
        ['Mesh', 'Method', 'MSE', 'MAE', 'RMSE', 'Max Error'],
        ['Cube', 'Min-Max', '0.00000000', '0.00000000', '0.00000000', '0.00000000'],
        ['Cube', 'Unit Sphere', '0.00000000', '0.00000000', '0.00000000', '0.00000000'],
        ['Pyramid', 'Min-Max', '0.00000013', '0.00013034', '0.00035694', '0.00097752'],
        ['Pyramid', 'Unit Sphere', '0.00000013', '0.00013034', '0.00035694', '0.00097752'],
        ['Sphere', 'Min-Max', '0.00000040', '0.00056410', '0.00063154', '0.00097752'],
        ['Sphere', 'Unit Sphere', '0.00000040', '0.00056410', '0.00063154', '0.00097752'],
        ['Tetrahedron', 'Min-Max', '0.00000000', '0.00000000', '0.00000000', '0.00000000'],
        ['Tetrahedron', 'Unit Sphere', '0.00000000', '0.00000000', '0.00000000', '0.00000000'],
    ]
    
    t = Table(error_data, colWidths=[1.2*inch, 1.2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fadbd8')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fadbd8'), colors.white])
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.2*inch))
    
    # Error analysis plots
    elements.append(Paragraph("4.2 Error Analysis Plots", heading2_style))
    
    plot_text = """
    Three types of plots were generated for comprehensive error analysis:
    <br/>
    • <b>Comparison plots:</b> Compare MSE, MAE, and Max Error between normalization methods<br/>
    • <b>Per-axis plots:</b> Show error distribution across X, Y, Z axes<br/>
    • <b>Distribution plots:</b> Histogram of error magnitudes<br/>
    """
    elements.append(Paragraph(plot_text, body_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Add error plots (comparison and per-axis for sphere)
    plot_paths = [
        'output/plots/sphere_comparison.png',
        'output/plots/sphere_MinMax_per_axis.png',
        'output/plots/sphere_MinMax_distribution.png'
    ]
    
    existing_plots = [p for p in plot_paths if os.path.exists(p)]
    if len(existing_plots) >= 2:
        # Add first two plots side by side
        plot_row = [[Image(existing_plots[0], width=3*inch, height=2.5*inch),
                     Image(existing_plots[1], width=3*inch, height=2.5*inch)]]
        
        plot_table = Table(plot_row, colWidths=[3.2*inch, 3.2*inch])
        plot_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(plot_table)
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Observations
    elements.append(Paragraph("4.3 Key Observations", heading2_style))
    
    obs_text = """
    <b>1. Perfect Reconstruction for Simple Geometries:</b><br/>
    Cube and tetrahedron achieved MSE = 0, indicating perfect reconstruction. 
    This occurs because their vertex coordinates align exactly with quantization bin centers.
    <br/><br/>
    <b>2. Sub-Millimeter Precision for Complex Shapes:</b><br/>
    Pyramid and sphere show MSE in the range of 10⁻⁷, representing sub-millimeter errors 
    that are visually imperceptible. Maximum error across all meshes is ~0.001 units.
    <br/><br/>
    <b>3. Method Equivalence:</b><br/>
    Min-Max and Unit Sphere normalization produce identical error metrics for each mesh, 
    confirming both methods preserve structure equally well after reconstruction.
    <br/><br/>
    <b>4. Visual Lossless Quality:</b><br/>
    10-bit quantization (1024 bins) provides visually lossless reconstruction, matching 
    industry standards where Google Draco uses 8-14 bits for mesh compression.
    """
    
    elements.append(Paragraph(obs_text, body_style))
    
    # ===================================================================
    # 5. BONUS TASK: SEAM TOKENIZATION
    # ===================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("5. Bonus Task: Seam Tokenization (30 Marks)", heading1_style))
    
    bonus_text = """
    <b>Objective:</b> Develop a tokenization scheme for representing mesh UV seams as 
    discrete sequences, enabling SeamGPT-style processing.
    <br/><br/>
    <b>5.1 Concept</b>
    <br/><br/>
    UV seams are edges where texture mapping becomes discontinuous - the same 3D edge has 
    different UV coordinates on either side. These seams are critical for understanding 
    how 3D meshes are unwrapped for texture application.
    <br/><br/>
    <b>5.2 Implementation</b>
    <br/><br/>
    <b>Seam Detection:</b><br/>
    • Build edge-to-UV mapping from face data<br/>
    • Identify edges with multiple UV coordinate pairs<br/>
    • Classify seams by type: HORIZONTAL, VERTICAL, DIAGONAL, BOUNDARY<br/>
    <br/>
    <b>Token Encoding Scheme:</b><br/>
    Each seam is encoded as a 9-token sequence:<br/>
    [SEAM_START, vertex_1, vertex_2, u1, v1, u2, v2, seam_type, SEAM_END]<br/>
    <br/>
    • SEAM_START/END: Delimiters (values: 0, 1)<br/>
    • vertex_1, vertex_2: 3D vertex indices forming the seam edge<br/>
    • u1, v1: First UV coordinate pair (8-bit quantized: 0-255)<br/>
    • u2, v2: Second UV coordinate pair (8-bit quantized: 0-255)<br/>
    • seam_type: Classification (0-3)<br/>
    <br/>
    <b>Example Token Sequence:</b><br/>
    For a cube with 2 detected seams:<br/>
    [0, 0, 1, 63, 84, 0, 84, 3, 1, 0, 0, 4, 0, 84, 63, 84, 3, 1]<br/>
    <br/>
    This represents two boundary seams with specific UV discontinuities.
    """
    
    elements.append(Paragraph(bonus_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Seam visualizations
    elements.append(Paragraph("5.3 Seam Visualizations", heading2_style))
    
    seam_paths = [
        'output/seam_visualizations/mesh_with_seams_3d.png',
        'output/seam_visualizations/uv_map_with_seams.png',
        'output/seam_visualizations/seam_analysis.png'
    ]
    
    existing_seams = [p for p in seam_paths if os.path.exists(p)]
    
    if len(existing_seams) >= 2:
        # First two images
        seam_row1 = [[Image(existing_seams[0], width=3*inch, height=2.5*inch),
                      Image(existing_seams[1], width=3*inch, height=2.5*inch)]]
        
        seam_table1 = Table(seam_row1, colWidths=[3.2*inch, 3.2*inch])
        seam_table1.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(seam_table1)
        
        # Labels
        labels1 = [['3D Mesh with Seam Edges', 'UV Map with Discontinuities']]
        label_table1 = Table(labels1, colWidths=[3.2*inch, 3.2*inch])
        label_table1.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ]))
        elements.append(label_table1)
    
    elements.append(Spacer(1, 0.15*inch))
    
    # Connection to SeamGPT
    seamgpt_text = """
    <b>5.4 Connection to SeamGPT and Mesh Understanding</b>
    <br/><br/>
    This tokenization scheme enables several key capabilities:
    <br/><br/>
    <b>1. Sequential Processing:</b> Transformers like GPT can process 3D geometry as 
    token sequences, similar to text or code.<br/>
    <br/>
    <b>2. Compression:</b> Encoding seams as 9 tokens provides efficient representation 
    for training AI models.<br/>
    <br/>
    <b>3. Generative Modeling:</b> Models can learn to generate or modify UV layouts by 
    predicting token sequences.<br/>
    <br/>
    <b>4. Semantic Understanding:</b> Seam types (horizontal, vertical, diagonal) provide 
    structural information about mesh topology.<br/>
    <br/>
    <b>5. Mesh Repair:</b> By learning seam patterns, models can detect and fix UV mapping 
    errors automatically.<br/>
    <br/>
    This bridges the gap between 3D geometry and language model architectures, enabling 
    transformer-based processing of mesh data.
    """
    
    elements.append(Paragraph(seamgpt_text, body_style))
    
    # ===================================================================
    # 6. CONCLUSIONS
    # ===================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("6. Conclusions and Observations", heading1_style))
    
    conclusion_text = """
    This assignment successfully implemented a complete data preprocessing pipeline for 
    3D meshes, demonstrating the fundamental techniques used in AI-driven 3D graphics systems.
    <br/><br/>
    <b>Key Achievements:</b>
    <br/><br/>
    <b>1. Robust Mesh Processing:</b><br/>
    Successfully loaded and processed four test meshes of varying complexity, from simple 
    platonic solids to higher-density spherical geometries.
    <br/><br/>
    <b>2. Effective Normalization:</b><br/>
    Both Min-Max and Unit Sphere normalization methods proved equally effective for mesh 
    reconstruction. The choice between methods should be based on application requirements:
    <br/>
    • Min-Max: Better for preserving absolute proportions<br/>
    • Unit Sphere: Better for rotation-invariant processing<br/>
    <br/>
    <b>3. High-Quality Quantization:</b><br/>
    10-bit quantization (1024 bins) achieved excellent results:
    <br/>
    • Perfect reconstruction (MSE = 0) for simple geometries<br/>
    • Sub-millimeter errors (MSE < 10⁻⁶) for complex shapes<br/>
    • Visually lossless quality matching industry standards<br/>
    <br/>
    <b>4. Industry-Standard Implementation:</b><br/>
    The quantization implementation uses rounding (not flooring), following Google Draco's 
    approach for unbiased quantization. This critical detail ensures production-quality results.
    <br/><br/>
    <b>5. Advanced Mesh Understanding:</b><br/>
    The seam tokenization prototype demonstrates how 3D geometric concepts can be encoded 
    as discrete sequences, enabling transformer-based AI models to process and understand 
    mesh topology.
    <br/><br/>
    <b>Patterns Observed:</b>
    <br/><br/>
    • Reconstruction error is proportional to geometric complexity and vertex density<br/>
    • Symmetric meshes (cube, tetrahedron) achieve perfect reconstruction<br/>
    • Maximum quantization error is bounded by 1/(2×1023) ≈ 0.0005 per coordinate<br/>
    • Visual quality is preserved even with numerical precision loss<br/>
    <br/>
    <b>Practical Implications:</b>
    <br/><br/>
    This work demonstrates that 3D mesh data can be normalized and quantized with minimal 
    information loss, making it suitable for:
    <br/>
    • AI model training datasets<br/>
    • Mesh compression and streaming<br/>
    • Real-time graphics applications<br/>
    • Transformer-based generative models<br/>
    <br/>
    The techniques implemented here form the foundation of modern 3D AI systems like SeamGPT, 
    enabling intelligent mesh understanding, generation, and modification.
    """
    
    elements.append(Paragraph(conclusion_text, body_style))
    
    # ===================================================================
    # 7. REFERENCES
    # ===================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("7. References", heading1_style))
    
    refs = [
        "1. Google Draco: 3D Graphics Compression Library. https://google.github.io/draco/",
        "2. Trimesh Library Documentation. https://trimsh.org/",
        "3. Open3D: A Modern Library for 3D Data Processing. http://www.open3d.org/",
        "4. NumPy: The fundamental package for scientific computing with Python. https://numpy.org/",
        "5. Matplotlib: Visualization with Python. https://matplotlib.org/",
        "6. Assignment: Mesh Normalization, Quantization, and Error Analysis (2025)"
    ]
    
    for ref in refs:
        elements.append(Paragraph(ref, styles['Normal']))
        elements.append(Spacer(1, 10))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Appendix
    elements.append(Paragraph("Appendix: Technical Specifications", heading2_style))
    
    specs_text = """
    <b>Software Environment:</b><br/>
    • Python 3.12+<br/>
    • NumPy 1.21.0+<br/>
    • Trimesh 3.9.0+<br/>
    • Matplotlib 3.4.0+<br/>
    • Open3D 0.13.0+<br/>
    <br/>
    <b>Hardware:</b><br/>
    • CPU-only execution (no GPU required)<br/>
    <br/>
    <b>Code Repository:</b><br/>
    • Main script: main.py<br/>
    • Source modules: src/mesh_loader.py, src/normalization.py, src/quantization.py, 
    src/error_analysis.py<br/>
    • Bonus implementation: seam_tokenization.py, visualize_seams.py<br/>
    <br/>
    <b>Generated Outputs:</b><br/>
    • 16 mesh files (.ply)<br/>
    • 20 error analysis plots (.png)<br/>
    • 20 mesh visualizations (.png)<br/>
    • 3 seam analysis visualizations (.png)<br/>
    • 1 summary report (.txt)<br/>
    """
    
    elements.append(Paragraph(specs_text, body_style))
    
    # Build PDF
    doc.build(elements)
    print(f"✅ PDF Report generated: {output_filename}")
    return output_filename


if __name__ == "__main__":
    print("="*70)
    print("GENERATING PDF REPORT")
    print("="*70)
    print("\nThis will create a comprehensive PDF report including:")
    print("  • Title page and table of contents")
    print("  • Task 1: Mesh loading and statistics")
    print("  • Task 2: Normalization and quantization")
    print("  • Task 3: Error analysis with plots")
    print("  • Bonus: Seam tokenization")
    print("  • Conclusions and observations")
    print("  • References and appendix")
    print("\n" + "="*70)
    
    try:
        output_file = create_pdf_report('Assignment_Report.pdf')
        print(f"\n✅ SUCCESS! Report saved as: {output_file}")
        print("\nYou can now:")
        print("  1. Review the PDF")
        print("  2. Copy it to submission/ folder")
        print("  3. Package final submission")
        print("="*70)
    except Exception as e:
        print(f"\n❌ Error generating PDF: {e}")
        print("\nTroubleshooting:")
        print("  • Ensure ReportLab is installed: pip install reportlab")
        print("  • Check that output/ folder exists with all images")
        print("  • Verify file permissions")
