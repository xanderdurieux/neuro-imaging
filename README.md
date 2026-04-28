# Neuroimaging visualization (Computer Graphics)

PyQt5 desktop application combining **VTK** for 3D volume and mesh visualization with **Qt** for controls. The course project is split into tabs; each tab is implemented as a self-contained widget under `src/`.

**Entry point:** `main.py` — run from the project root so imports resolve (`python main.py`).

---

## Quick start

1. **Python** 3.10+ recommended (matches type-hint syntax used in the codebase).

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Stack: `vtk`, `PyQt5`, `numpy`, `matplotlib` (Task 2 uses Matplotlib embedded in Qt).

3. **Data** — Download and unpack the course imaging data bundle so the `Neuroimaging_visualization_project/` directory is available in your project root.

   ```
   Neuroimaging_visualization_project/
   ├── head_with_lesion.vtk              # Tasks 1 & 2 — PATHS["head"]
   ├── head_with_lesion.vti              # optional; same volume, XML format
   ├── vessels_data.vtk                  # Task 4 — PATHS["vessels"]
   ├── vessels_data.vti
   ├── main.pdf                          # course PDF (not read by the app)
   ├── MARCHING CUBES/                   # reference VTK snippets
   ├── Useful_VTK_examples/
   └── DSA of AVM examples/              # Task 3 — PATHS["dsa_dir"]
      ├── vdm01_1.vtk                   # default series 1 — PATHS["dsa1"]
      ├── vdm02_1.vtk                   # default series 2 — PATHS["dsa2"]
      ├── bb01_1.vtk
      ├── bb01_2.vtk
      ├── bb02_1.vtk
      ├── bb02_2.vtk                    # … additional bb*/embo* .vtk / .vti …
      ├── BB01/
      │   └── *.png                     # PNG frame sequence
      ├── BB02/
      │   ├── bb0000.png
      │   ├── bb0001.png
      │   └── …                       # one PNG per time frame
      ├── Embo03/
      │   ├── embo0000.png
      │   └── …
      └── …                           # other case folders (BB03, Embo01, …)
   ```

   If a path in `PATHS` is missing, the status bar lists it (orange). After cloning, unpack the course imaging bundle so the `Neuroimaging_visualization_project/` tree matches the above.

4. **Run**

   ```bash
   python main.py
   ```

---

## Project layout

| Path | Role |
|------|------|
| `main.py` | `QApplication`, `QMainWindow`, `QTabWidget`; wires tabs and calls `check_data_files()`. VTK output is sent away from GUI popups (`vtkOutputWindow.NEVER`). |
| `src/utils.py` | Shared **data paths**, VTK readers, marching-cubes helper (`build_isosurface`), renderer/axes helpers, lookup tables, `numpy` ↔ `vtkImageData` conversion. |
| `src/task1_basic_viz.py` | **Task 2.1** — `Task1Widget`: orthogonal slice planes + labeled-tissue marching cubes. |
| `src/task2_eeg_viz.py` | **Task 2.2** — `Task2Widget`: brain surface, electrode placement, IDW coloring, timers, Matplotlib signal plots. |
| `src/task3_dsa_viz.py` | **Task 2.3** — `Task3Widget`: DSA-style visualization. |
| `src/task4_mip.py` | **Task 2.4** — `Task4Widget`: MIP / volume slicing  |
| `requirements.txt` | Pinned minimum versions for reproducible installs. |

Module docstrings at the top of each `task*.py` file state the assignment requirements that module implements.

---

## Task 1 — Basic visualization (`Task1Widget`)

**File:** `src/task1_basic_viz.py`

**Data:** `head_with_lesion.vtk` (structured points, `COLOR_SCALARS` as labels).

**Pipeline (high level):**

- `vtkStructuredPointsReader` loads the volume.
- Three `vtkImagePlaneWidget` instances (sagittal / transversal / coronal) for slice browsing; Qt sliders and keyboard shortcuts adjust slice index.
- For each tissue band, `build_isosurface` in `utils.py` runs marching cubes (`vtkContourFilter`), optional smoothing, normals, and produces a `vtkActor`. Iso-values in `ISO_PRESETS` match the documented label midpoints for this dataset.
- Checkboxes and keys `1`–`5` toggle mesh visibility.

---

## Task 2 — EEG-style visualization (`Task2Widget`)

**File:** `src/task2_eeg_viz.py`

**Data:** same `head_with_lesion.vtk` as Task 1.

**Pipeline (high level):**

- Brain (and skin) surfaces via contouring at `BRAIN_ISO` / `SKIN_ISO`, with smoothing suitable for picking and display.
- Up to `N_ELECTRODES` (8) positions chosen with `vtkCellPicker`, restricted to the brain actor; glyphs (`vtkGlyph3D` + sphere source) mark electrodes.
- Scalar values on brain vertices: **inverse-distance weighting (IDW)** with \(w_i = 1/d_i^2\), implemented with NumPy over mesh points (see module docstring).
- `QTimer` periodically re-randomizes electrode values among `ELECTRODE_VALUES` and refreshes colors.
- Right-hand side: Matplotlib `FigureCanvas` with one subplot per electrode, showing recent samples; window length `T` is configurable.

---

## Task 3 — DSA visualization (`Task3Widget`)

**File:** `src/task3_dsa_viz_threshold.py`

**Data:** PNG sequences from `DSA of AVM examples/` (each subfolder contains two temporal series of images).

**Pipeline:**

- `vtkPNGReader` loads each frame of the DSA sequence. The pipeline automatically detects the dominant frame size and discards inconsistent frames (e.g., in Embo datasets) to ensure a valid NumPy stack (N, H, W). It also splits the series into distinct "views" by detecting intensity jumps (removing anomalous/reference frames).
- Contrast is isolated by calculating the absolute difference between each pixel and the median intensity of the stack.
- A time map is computed per pixel using **argmax** or **intensity-weighted average** to encode contrast arrival.
- The map is smoothed with a Gaussian filter, color-mapped (Red-to-Blue) via `vtkImageMapToColors`, and rendered using `vtkImageActor`. Pixels without signal are masked to dark gray.
- `vtkCellPicker` captures pixel coordinates. Left-clicking opens a separate dialog with a Matplotlib plot showing the contrast flow curve (mean intensity over a configurable neighborhood).

---

## Task 4 — MIP slicing (`Task4Widget`)

**File:** `src/task4_mip.py`

**Data:** `vessels_data.vtk`.

**Pipeline (high level):**

- `vtkStructuredPointsReader` loads the volume and the scalar array is cached as a NumPy array of shape `(Z, Y, X)` for efficient slab extraction.
- For each of the three orthogonal views (Axial / Sagittal / Coronal) a slab of slices is extracted with NumPy indexing and projected to a 2-D image via `np.max` along the slab axis (this is the Maximum Intensity Projection step).
- The resulting 2-D array is wrapped as `vtkImageData` via `numpy_to_vtk_image` (utils.py) and passed through `vtkImageMapToWindowLevelColors` for brightness/contrast adjustment before display with `vtkImageActor`.
- A global slab-thickness slider controls what percentage of the total slices in each direction forms the projection slab (default: 10 %).
- Window and Level sliders adjust the grey-scale mapping across all three views simultaneously.

---

## Shared utilities (`src/utils.py`)

Worth scanning as the **single place** for:

- `PATHS` / `check_data_files()`
- `read_vtk_structured_points` / `read_vtk_structured_points_as_image`
- `build_isosurface` (marching cubes + smoothing + mapper/actor)
- `make_renderer`, `add_axes`, colormap helpers, `numpy_to_vtk_image`
