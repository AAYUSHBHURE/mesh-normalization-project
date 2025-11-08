# 🎓 3D Mesh Normalization, Quantization, and Error Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)]()

## 📋 Assignment Overview

This project implements **complete data preprocessing for 3D meshes**, a fundamental step in AI-driven 3D graphics systems like SeamGPT. The implementation covers:

- ✅ **Normalization**: Bringing mesh coordinates into standard ranges (Min-Max, Unit Sphere)
- ✅ **Quantization**: Discretizing coordinates into 1024 bins
- ✅ **Error Analysis**: Measuring reconstruction accuracy with MSE, MAE, RMSE
- ✅ **Visualization**: Generating plots and 3D visualizations
- ✅ **Bonus Tasks**: Rotation-invariant normalization and adaptive quantization

**Total Marks: 100 + 30 Bonus**

---

## 🚀 Quick Start - How to Execute

### Prerequisites
- Python 3.8 or higher installed
- Virtual environment activated (recommended)

### Step 1: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Array operations and numerical computing
- `trimesh` - Mesh loading and processing
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing utilities

### Step 2: Prepare Mesh Data
The project includes pre-generated sample meshes. To view or regenerate:

```bash
# Option A: Use existing meshes in data/ folder (already included)
# cube.obj, pyramid.obj, sphere.obj, tetrahedron.obj

# Option B: Regenerate sample meshes (optional)
python generate_sample_meshes.py

# Option C: Add your own .obj meshes to data/ folder
```

### Step 3: Run Main Pipeline
Execute the complete normalization, quantization, and error analysis:

```bash
# Run all tasks (1, 2, 3) and generate all outputs
python main.py
```

**This will:**
- ✅ Load all meshes from `data/` folder
- ✅ Compute statistics (Task 1)
- ✅ Normalize using Min-Max and Unit Sphere methods (Task 2)
- ✅ Quantize to 1024 bins (Task 2)
- ✅ Reconstruct and compute errors (Task 3)
- ✅ Generate 20+ plots and visualizations
- ✅ Save all results to `output/` folder
- ✅ Create summary report

**Expected output:**
```
output/
├── meshes/              # 16 .ply files (quantized + reconstructed)
├── plots/               # 20 .png error analysis plots
├── visualizations/      # 20 .png mesh renders
└── SUMMARY_REPORT.txt   # Complete metrics table
```

### Step 4: Run Bonus Task - Seam Tokenization (Optional)

```bash
# Execute seam tokenization demonstration
python seam_tokenization.py

# Generate seam visualizations
python visualize_seams.py
```

**This creates:**
```
output/seam_visualizations/
├── mesh_with_seams_3d.png    # 3D mesh with seams highlighted
├── uv_map_with_seams.png     # UV layout with discontinuities
└── seam_analysis.png          # Token encoding details
```

### Step 5: Create Submission Package

```bash
# Windows PowerShell
Compress-Archive -Path submission\* -DestinationPath submission.zip -Force

# Linux/Mac
zip -r submission.zip submission/
```

### 🎯 Complete Execution Workflow

**Full pipeline from start to finish:**

```bash
# 1. Activate virtual environment (if using)
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation (optional but recommended)
python test_installation.py

# 4. Run core tasks (1, 2, 3)
python main.py

# 5. Run bonus task - seam tokenization
python seam_tokenization.py
python visualize_seams.py

# 6. Generate visualizations for report (optional)
python visualize_states.py

# 7. Review outputs
# - Check output/meshes/ for processed meshes
# - Check output/plots/ for error charts
# - Check output/visualizations/ for mesh renders
# - Check output/SUMMARY_REPORT.txt for metrics

# 8. Create PDF report using generated assets
# (Use images from output/ folders in your PDF)

# 9. Package submission
# Copy your REPORT.pdf to submission/ folder
Compress-Archive -Path submission\* -DestinationPath submission.zip -Force
```

### 🔍 Individual Module Testing

Test specific components independently:

```bash
# Test mesh loading only
python -c "from src.mesh_loader import load_mesh; print(load_mesh('data/cube.obj')[0].shape)"

# Test normalization only
python -c "from src.normalization import MinMaxNormalizer; import numpy as np; n = MinMaxNormalizer(); print(n.normalize(np.array([[1,2,3]])))"

# Test quantization only
python -c "from src.quantization import quantize_vertices; import numpy as np; print(quantize_vertices(np.array([[0.5, 0.5, 0.5]]), 1024))"

# Run error analysis demo
python src/error_analysis.py
```

---

## 📁 Project Structure

```
mesh-normalization-project/
├── 📂 data/                    # Place your .obj mesh files here
│   └── README.txt
├── 📂 src/                     # Source code modules
│   ├── mesh_loader.py         # Task 1: Load and inspect meshes
│   ├── normalization.py       # Task 2: Normalization methods
│   ├── quantization.py        # Task 2: Quantization (1024 bins)
│   ├── error_analysis.py      # Task 3: Error metrics and plotting
│   ├── visualization.py       # 3D visualization with Open3D
│   └── bonus_tasks.py         # Optional bonus features
├── 📂 output/                  # Generated results (auto-created)
│   ├── meshes/                # Normalized and quantized meshes
│   ├── plots/                 # Error analysis plots
│   └── visualizations/        # Mesh screenshots
├── 📄 main.py                 # ⭐ Main execution script
├── 📄 generate_sample_meshes.py  # Create test meshes
├── 📄 test_installation.py    # Verify setup
├── 📄 requirements.txt        # Python dependencies
├── 📄 setup.bat / setup.sh    # Automated setup scripts
├── 📄 GET_STARTED.md          # Getting started guide
├── 📄 QUICKSTART.md           # Quick reference
└── 📄 README.md               # This file
```

---

## 🎯 Tasks Implemented

### ✅ Task 1: Load and Inspect Mesh (20 Marks)

**Implementation:** `src/mesh_loader.py`

- Loads `.obj` files using Trimesh library
- Extracts vertex coordinates as NumPy arrays
- Computes statistics: count, min, max, mean, std deviation per axis
- Supports batch loading of multiple meshes
- Optional visualization support

**Key Functions:**
- `load_mesh(filepath)` - Load single mesh
- `load_all_meshes(folder)` - Load all meshes from folder
- `compute_statistics(vertices)` - Calculate mesh statistics
- `print_statistics(stats, name)` - Display formatted statistics

---

### ✅ Task 2: Normalize and Quantize (40 Marks)

**Implementation:** `src/normalization.py`, `src/quantization.py`

#### Normalization Methods:

**1. Min-Max Normalization**
```python
normalized = (vertices - v_min) / (v_max - v_min)
```
- Scales coordinates to [0, 1] range
- Preserves aspect ratio and shape
- Best for uniform scaling

**2. Unit Sphere Normalization**
```python
centered = vertices - center
normalized = centered / max_distance
```
- Centers mesh at origin
- Scales to fit in unit sphere (radius = 1)
- Good for rotation-invariant processing

#### Quantization:
- Discretizes normalized coordinates into **1024 bins**
- Formula: `quantized = floor(normalized × 1023)`
- Saves quantized meshes as `.ply` files

**Key Functions:**
- `MinMaxNormalizer()` - Min-Max normalization class
- `UnitSphereNormalizer()` - Unit Sphere normalization class
- `quantize_mesh(vertices, normalizer, bins)` - Complete quantization pipeline
- `dequantize_mesh(quantized, params, normalizer)` - Reconstruction pipeline

---

### ✅ Task 3: Reconstruction and Error Analysis (40 Marks)

**Implementation:** `src/error_analysis.py`

#### Error Metrics:
- **MSE** (Mean Squared Error): `mean((original - reconstructed)²)`
- **MAE** (Mean Absolute Error): `mean(|original - reconstructed|)`
- **RMSE** (Root Mean Squared Error): `sqrt(MSE)`
- **Max Error**: Maximum absolute difference
- **Per-Axis Analysis**: Separate metrics for X, Y, Z

#### Visualizations:
1. **Error Comparison Plot** - Compare normalization methods
2. **Per-Axis Error Plot** - MSE and MAE for each axis
3. **Error Distribution** - Histograms showing error spread

**Key Functions:**
- `compute_all_metrics(original, reconstructed)` - All error metrics
- `plot_error_comparison(error_results, path)` - Method comparison chart
- `plot_per_axis_errors(metrics, method, path)` - Axis-wise analysis
- `plot_error_distribution(original, reconstructed, path)` - Error histograms

---

### 🌟 Bonus Tasks (30 Marks)

**Implementation:** `seam_tokenization.py`, `visualize_seams.py`

#### Seam Tokenization for SeamGPT-Style Processing

**Overview:**
Represents mesh UV seams (texture discontinuities) as discrete token sequences that can be processed by transformer models like GPT.

**Features:**
- **Seam Detection**: Identifies UV discontinuities where texture mapping breaks
- **Token Encoding**: Converts seams to 9-token sequences: `[START, v1, v2, u1, v1, u2, v2, type, END]`
- **8-bit Quantization**: UV coordinates quantized to 256 levels
- **Seam Classification**: Categorizes as HORIZONTAL, VERTICAL, DIAGONAL, or BOUNDARY
- **Round-trip Encoding**: Encodes and decodes with minimal precision loss

**How to Run:**
```bash
# Run tokenization demonstration
python seam_tokenization.py

# Generate seam visualizations
python visualize_seams.py
```

**Outputs:**
```
output/seam_visualizations/
├── mesh_with_seams_3d.png       # 3D mesh with red seam edges
├── uv_map_with_seams.png        # UV layout with discontinuities
└── seam_analysis.png            # Statistics and token encoding
```

**Key Classes:**
- `SeamTokenizer()` - Main tokenization engine
- `SeamEdge()` - Data structure for seam representation
- `TokenType` - Enum for token categories
- `SeamType` - Enum for seam classifications

**Connection to SeamGPT:**
This tokenization enables:
1. **Sequential Processing**: Transformers can process 3D geometry as token sequences
2. **Compression**: 9 tokens per seam provides efficient representation
3. **Generative Modeling**: Models can learn to generate/modify UV layouts
4. **Semantic Understanding**: Seam types provide topology information
5. **Mesh Repair**: AI can detect and fix UV mapping errors

---

## 📊 Generated Outputs

When you run the project, outputs are organized in the `output/` folder:

### Core Task Outputs

#### Meshes (`output/meshes/`) - 16 files
```
cube_MinMax_quantized.ply              # Cube normalized with Min-Max, quantized
cube_MinMax_reconstructed.ply          # Reconstructed from quantized
cube_UnitSphere_quantized.ply          # Cube normalized with Unit Sphere
cube_UnitSphere_reconstructed.ply      # Reconstructed from quantized
pyramid_MinMax_quantized.ply
pyramid_MinMax_reconstructed.ply
pyramid_UnitSphere_quantized.ply
pyramid_UnitSphere_reconstructed.ply
sphere_MinMax_quantized.ply
sphere_MinMax_reconstructed.ply
sphere_UnitSphere_quantized.ply
sphere_UnitSphere_reconstructed.ply
tetrahedron_MinMax_quantized.ply
tetrahedron_MinMax_reconstructed.ply
tetrahedron_UnitSphere_quantized.ply
tetrahedron_UnitSphere_reconstructed.ply
```

#### Error Analysis Plots (`output/plots/`) - 20 files
```
# Comparison plots (4 meshes)
cube_comparison.png
pyramid_comparison.png
sphere_comparison.png
tetrahedron_comparison.png

# Per-axis error breakdown (8 plots: 4 meshes × 2 methods)
cube_MinMax_per_axis.png
cube_UnitSphere_per_axis.png
pyramid_MinMax_per_axis.png
pyramid_UnitSphere_per_axis.png
sphere_MinMax_per_axis.png
sphere_UnitSphere_per_axis.png
tetrahedron_MinMax_per_axis.png
tetrahedron_UnitSphere_per_axis.png

# Error distributions (8 plots: 4 meshes × 2 methods)
cube_MinMax_distribution.png
cube_UnitSphere_distribution.png
pyramid_MinMax_distribution.png
pyramid_UnitSphere_distribution.png
sphere_MinMax_distribution.png
sphere_UnitSphere_distribution.png
tetrahedron_MinMax_distribution.png
tetrahedron_UnitSphere_distribution.png
```

#### Mesh Visualizations (`output/visualizations/`) - 20 images
```
cube/
  cube_original.png                    # Original mesh
  cube_MinMax_normalized.png           # After Min-Max normalization
  cube_MinMax_reconstructed.png        # After quantization + reconstruction
  cube_UnitSphere_normalized.png
  cube_UnitSphere_reconstructed.png

pyramid/
  pyramid_original.png
  pyramid_MinMax_normalized.png
  pyramid_MinMax_reconstructed.png
  pyramid_UnitSphere_normalized.png
  pyramid_UnitSphere_reconstructed.png

sphere/
  sphere_original.png
  sphere_MinMax_normalized.png
  sphere_MinMax_reconstructed.png
  sphere_UnitSphere_normalized.png
  sphere_UnitSphere_reconstructed.png

tetrahedron/
  tetrahedron_original.png
  tetrahedron_MinMax_normalized.png
  tetrahedron_MinMax_reconstructed.png
  tetrahedron_UnitSphere_normalized.png
  tetrahedron_UnitSphere_reconstructed.png
```

#### Summary Report (`output/`)
```
SUMMARY_REPORT.txt                     # Complete metrics table for all meshes
```

### Bonus Task Outputs

#### Seam Visualizations (`output/seam_visualizations/`) - 3 files
```
mesh_with_seams_3d.png                 # 3D mesh with seam edges highlighted (red)
uv_map_with_seams.png                  # UV layout showing discontinuities
seam_analysis.png                      # Statistics, token encoding dashboard
```

### Total Files Generated
- **59 files** in `output/` folder
- **16 meshes** (.ply format)
- **40 visualizations** (.png images)
- **1 text report** (SUMMARY_REPORT.txt)
- **2 bonus visualizations** (seam analysis)

---

## 💻 Usage Examples

### Basic Usage - Complete Pipeline
```bash
# Run all tasks and generate all outputs
python main.py
```

**What this does:**
1. Loads all meshes from `data/` folder
2. Computes statistics (vertices, bounds, mean, std)
3. Normalizes using Min-Max and Unit Sphere
4. Quantizes to 1024 bins
5. Reconstructs meshes
6. Computes error metrics (MSE, MAE, RMSE)
7. Generates comparison plots
8. Saves meshes and reports

### Bonus Task - Seam Tokenization
```bash
# Demonstrate seam tokenization
python seam_tokenization.py

# Generate visualizations
python visualize_seams.py
```

**What this does:**
1. Creates example cube with UV mapping
2. Detects UV seams (discontinuities)
3. Encodes seams as token sequences
4. Decodes tokens back to seams
5. Generates 3D and UV visualizations
6. Shows connection to SeamGPT research

### Generate Mesh Visualizations
```bash
# Create before/after mesh renders
python visualize_states.py
```

**Output:** 20 PNG files showing original, normalized, and reconstructed meshes

### Test Individual Modules
```python
# Test mesh loading
python -c "from src.mesh_loader import load_mesh; v, f = load_mesh('data/cube.obj'); print(f'Vertices: {len(v)}')"

# Test normalization
python -c "from src.normalization import MinMaxNormalizer; import numpy as np; n = MinMaxNormalizer(); print(n.normalize(np.array([[0, 1, 2]])))"

# Test quantization
python -c "from src.quantization import quantize_vertices, dequantize_vertices; import numpy as np; q = quantize_vertices(np.array([[0.5, 0.5, 0.5]]), 1024); print(q)"
```

### Custom Pipeline
```python
from src.mesh_loader import load_mesh
from src.normalization import MinMaxNormalizer
from src.quantization import quantize_mesh, dequantize_mesh
from src.error_analysis import compute_all_metrics

# Load mesh
vertices, faces = load_mesh('data/bunny.obj')

# Normalize and quantize
normalizer = MinMaxNormalizer()
quantized, params = quantize_mesh(vertices, normalizer, bins=1024)

# Reconstruct
reconstructed = dequantize_mesh(quantized, params, normalizer)

# Analyze errors
metrics = compute_all_metrics(vertices, reconstructed)
print(f"MSE: {metrics['mse']:.8f}")
print(f"MAE: {metrics['mae']:.8f}")
```

---

## 📝 Creating Your Final Report

Use the generated outputs to create your PDF submission:

### Report Structure

**1. Introduction**
- Assignment context (SeamGPT, 3D data preprocessing)
- Objectives and scope

**2. Task 1: Mesh Loading (20 marks)**
- Methodology: How meshes are loaded
- Statistics: Include printed statistics from console
- Screenshots or tables showing mesh information

**3. Task 2: Normalization & Quantization (40 marks)**
- **Normalization Methods:**
  - Explain Min-Max normalization (formula and purpose)
  - Explain Unit Sphere normalization (formula and purpose)
- **Quantization:**
  - Explain 1024-bin quantization process
  - Show quantization/dequantization formulas
- **Results:** Include saved mesh files

**4. Task 3: Error Analysis (40 marks)**
- **Error Metrics:**
  - MSE, MAE, RMSE values for each method
  - Per-axis error breakdown
- **Visualizations:**
  - Include comparison plots
  - Include per-axis error plots
  - Include error distribution histograms
- **Analysis:**
  - Which method gives lower error?
  - Why? (Min-Max typically preserves structure better)
  - Are errors uniform across axes?

**5. Conclusion**
- Key findings and observations
- Best normalization method for this use case
- Lessons learned about data preprocessing

### Key Questions to Answer
- ✅ Which normalization method preserves mesh structure better?
- ✅ How much information is lost during quantization?
- ✅ Are reconstruction errors uniform across X, Y, Z axes?
- ✅ What patterns appear in error distributions?
- ✅ Why might one method outperform the other?

---

## 🔧 Dependencies

### Required Packages
```
numpy>=1.21.0          # Numerical operations
trimesh>=3.9.0         # Mesh loading and manipulation
matplotlib>=3.4.0      # Plotting and visualization
scipy>=1.7.0           # Scientific computing
```

### Optional Packages
```
open3d>=0.13.0         # 3D visualization (recommended)
reportlab>=3.6.0       # PDF generation
Pillow>=8.3.0          # Image processing
```

### Installation
```bash
pip install -r requirements.txt
```

---

## 🧪 Testing Your Setup

Before running the main project:

```bash
python test_installation.py
```

This verifies:
- ✅ All required packages are installed
- ✅ Project modules can be imported
- ✅ Basic functionality works correctly

---

## 🐛 Troubleshooting

### Import Errors
**Problem:** `Import "src.mesh_loader" could not be resolved`

**Solution:** This is normal before installation. Run:
```bash
pip install -r requirements.txt
```

### No Mesh Files Found
**Problem:** `⚠ No .obj files found in 'data/' folder`

**Solutions:**
1. Generate sample meshes: `python generate_sample_meshes.py`
2. Download meshes from:
   - http://graphics.stanford.edu/data/3Dscanrep/
   - https://github.com/alecjacobson/common-3d-test-models
3. Place `.obj` files in the `data/` folder

### Open3D Visualization Not Working
**Problem:** Visualization fails or crashes

**Solution:** The program will still work and generate all meshes and plots. Visualization is optional. You can:
- View meshes in Blender or MeshLab
- Use generated plots for analysis
- Skip visualization entirely

### Module Not Found
**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Ensure you're in the project directory:
```bash
cd C:\Users\bhure\mesh-normalization-project
python main.py
```

---

## 📚 Technical Documentation

### Normalization Formulas

#### Min-Max Normalization
Scales coordinates to [0, 1] range:

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Denormalization:**

$$x = x' \cdot (x_{max} - x_{min}) + x_{min}$$

#### Unit Sphere Normalization
Centers at origin and scales to unit radius:

$$x' = \frac{x - \mu}{r_{max}}$$

where $\mu$ is the centroid and $r_{max}$ is the maximum distance from center.

**Denormalization:**

$$x = x' \cdot r_{max} + \mu$$

### Quantization Process

**Forward (Quantization):**

$$q = \lfloor x' \times (n_{bins} - 1) \rfloor$$

where $x' \in [0, 1]$ and $n_{bins} = 1024$

**Inverse (Dequantization):**

$$x'' = \frac{q}{n_{bins} - 1}$$

### Error Metrics

**Mean Squared Error (MSE):**

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$$

**Mean Absolute Error (MAE):**

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |x_i - \hat{x}_i|$$

**Root Mean Squared Error (RMSE):**

$$RMSE = \sqrt{MSE}$$

---

## 🎯 Submission Checklist

### Code & Outputs
- [ ] All Python scripts (`.py` files)
- [ ] Generated meshes (`output/meshes/`)
- [ ] Error analysis plots (`output/plots/`)
- [ ] Summary report (`output/SUMMARY_REPORT.txt`)

### Documentation
- [ ] README explaining how to run code
- [ ] Comments in code explaining key sections
- [ ] Requirements file with dependencies

### Final Report (PDF)
- [ ] Task 1: Statistics and methodology
- [ ] Task 2: Normalization and quantization results
- [ ] Task 3: Error analysis with plots
- [ ] Comparison of normalization methods
- [ ] Written conclusion (5-10 lines)
- [ ] Observations and insights

### Submission Format
Create a ZIP file containing:
```
submission.zip
├── code/                    # All Python scripts
├── output/                  # Meshes, plots, reports
├── README.md                # How to run
└── REPORT.pdf              # Final written report
```

---

## 💡 Tips for Success

### Code Quality
- ✅ Use meaningful variable names
- ✅ Add comments explaining complex sections
- ✅ Follow consistent formatting
- ✅ Handle errors gracefully

### Analysis Quality
- ✅ Compare both normalization methods
- ✅ Explain why differences occur
- ✅ Use visualizations to support claims
- ✅ Provide quantitative comparisons (MSE, MAE)

### Report Quality
- ✅ Clear structure with sections
- ✅ Include all required plots
- ✅ Explain formulas and methods
- ✅ Write in your own words
- ✅ Provide thoughtful analysis

### Common Observations
- **Min-Max** typically has lower reconstruction error
- **Unit Sphere** is better for rotation-invariant tasks
- Quantization causes uniform random error ≈ 1/(2×1023)
- Errors are usually uniform across axes for symmetric meshes

---

## 📖 Additional Resources

### Mesh File Sources
- [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/)
- [Common 3D Test Models](https://github.com/alecjacobson/common-3d-test-models)
- [Thingi10K Dataset](https://ten-thousand-models.appspot.com/)
- [Princeton Shape Benchmark](http://shape.cs.princeton.edu/benchmark/)

### Learning Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [Trimesh Documentation](https://trimsh.org/)
- [Open3D Tutorial](http://www.open3d.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

### Related Concepts
- 3D Mesh Processing
- Data Normalization Techniques
- Quantization Methods
- Error Analysis and Metrics
- Computer Graphics Fundamentals

---

## 🏆 Grading Breakdown (100 + 30 Bonus)

| Task | Marks | Criteria |
|------|-------|----------|
| **Task 1: Load & Inspect** | 20 | Correct loading, statistics, visualization |
| **Task 2: Normalize & Quantize** | 40 | Two methods implemented, quantization works, meshes saved |
| **Task 3: Error Analysis** | 40 | Reconstruction works, error metrics computed, plots generated, written analysis |
| **Bonus: Advanced Features** | +30 | Rotation invariance, adaptive quantization, thorough analysis |
| **Total** | 100-130 | |

---

## 🤝 Contributing & Feedback

This is an educational project. If you find issues or have suggestions:
1. Review the code and understand what it does
2. Test modifications with sample meshes
3. Document your findings

---

## 📄 License

This project is created for educational purposes as part of a 3D graphics and AI assignment.

**Academic Integrity:**
- Use this as a learning tool
- Understand each component
- Write your own analysis
- Cite appropriately if required

---

## 👨‍💻 About This Project

**Assignment:** Mesh Normalization, Quantization, and Error Analysis  
**Context:** Preparing 3D data for AI models like SeamGPT  
**Focus:** Data preprocessing fundamentals  
**Total Marks:** 100 (+ 30 bonus)

**Key Learning Outcomes:**
1. Understand 3D mesh data structures
2. Implement normalization techniques
3. Apply quantization for data compression
4. Measure and analyze reconstruction errors
5. Compare different preprocessing methods

---

## 🚀 Quick Commands Reference

```bash
# ============================================
# SETUP AND INSTALLATION
# ============================================

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py


# ============================================
# CORE EXECUTION
# ============================================

# Run complete pipeline (Tasks 1, 2, 3)
python main.py

# Expected output:
# - Loads 4 meshes from data/
# - Computes statistics
# - Normalizes (Min-Max, Unit Sphere)
# - Quantizes to 1024 bins
# - Reconstructs meshes
# - Computes errors (MSE, MAE, RMSE)
# - Generates 40 plots
# - Saves 16 meshes
# - Creates summary report


# ============================================
# BONUS TASK - SEAM TOKENIZATION
# ============================================

# Run seam tokenization demonstration
python seam_tokenization.py

# Generate seam visualizations
python visualize_seams.py

# Expected output:
# - Detects 2 UV seams in example cube
# - Encodes seams as token sequences
# - Creates 3 visualization images


# ============================================
# VISUALIZATIONS
# ============================================

# Generate mesh state visualizations
python visualize_states.py

# Expected output:
# - 20 PNG images showing original/normalized/reconstructed meshes


# ============================================
# OPENING OUTPUTS
# ============================================

# Windows PowerShell
Invoke-Item output\plots\cube_comparison.png
Invoke-Item output\visualizations\sphere\sphere_original.png
Invoke-Item output\seam_visualizations\mesh_with_seams_3d.png
Invoke-Item output\SUMMARY_REPORT.txt

# Linux/Mac
open output/plots/cube_comparison.png
open output/visualizations/sphere/sphere_original.png
open output/seam_visualizations/mesh_with_seams_3d.png
open output/SUMMARY_REPORT.txt


# ============================================
# TESTING INDIVIDUAL MODULES
# ============================================

# Test mesh loading
python -c "from src.mesh_loader import load_mesh; v, f = load_mesh('data/cube.obj'); print(f'Loaded {len(v)} vertices')"

# Test normalization
python -c "from src.normalization import MinMaxNormalizer; import numpy as np; n = MinMaxNormalizer(); print(n.normalize(np.array([[0, 1, 2]])))"

# Test quantization
python -c "from src.quantization import quantize_vertices; import numpy as np; q = quantize_vertices(np.array([[0.5, 0.5, 0.5]]), 1024); print(q)"


# ============================================
# SUBMISSION PACKAGING
# ============================================

# Create submission ZIP (Windows)
Compress-Archive -Path submission\* -DestinationPath submission.zip -Force

# Create submission ZIP (Linux/Mac)
zip -r submission.zip submission/

# Verify submission contents
# Windows
Expand-Archive -Path submission.zip -DestinationPath temp_verify
dir temp_verify

# Linux/Mac
unzip -l submission.zip


# ============================================
# TROUBLESHOOTING
# ============================================

# Check Python version (need 3.8+)
python --version

# List installed packages
pip list

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Clear and regenerate outputs
rmdir output /s /q  # Windows
rm -rf output       # Linux/Mac
python main.py
```

---

## 📞 Support

For questions about the assignment:
- Review the assignment PDF
- Check `GET_STARTED.md` for setup help
- Check `QUICKSTART.md` for quick reference
- Review code comments for implementation details

---

## ✨ Acknowledgments

- **Trimesh** for mesh loading capabilities
- **Open3D** for 3D visualization
- **NumPy** for numerical computing
- **Matplotlib** for plotting

---

**Ready to start?** Run `setup.bat` (Windows) or `setup.sh` (Linux/Mac) to begin! 🎓

---

*Last Updated: 2025-11-07*
