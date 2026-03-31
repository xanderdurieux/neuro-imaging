"""
Task 2.3 – DSA (Digital Subtraction Angiography) Visualization
================================================================
DSA is a technique where frames are acquired at regular
intervals so that contrast agent propagation through blood vessels can be
tracked over time.

Two coloring strategies map each pixel to a single representative frame
index, which is then rendered through a red-to-blue lookup table:

    Argmax        — frame of peak intensity (sharpest boundaries)
    Weighted avg  — intensity-weighted mean of frame indices (smoother,
                    encodes the whole visible interval)

Left-clicking any pixel opens a separate window showing the contrast flow curve at that location, 
averaged over a square neighbourhood of adjustable radius. 
Multiple curves can be selected and compared in the same plot.
"""

import os
import glob
import vtk
import numpy as np
from vtkmodules.util import numpy_support as vtk_numpy
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QSlider,
    QSplitter, QSizePolicy, QButtonGroup, QRadioButton,
    QDialog, QFileDialog,
)
from PyQt5.QtCore import Qt

from .utils import PATHS, make_renderer, make_time_lut, numpy_to_vtk_image


# Fraction of the global maximum used as the background exclusion threshold
BACKGROUND_THRESHOLD_PCT = 0.05

# Default neighbourhood radius for flow-curve averaging (pixels)
DEFAULT_NEIGHBOUR_RADIUS = 3

CHART_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


class FlowCurveDialog(QDialog):
    """
    Separate window that displays one or more contrast flow curves.
    The window stays open and is redrawn whenever new picked points are added.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Contrast flow curves")
        self.resize(700, 460)
        self.setStyleSheet("""
            QDialog {
                background-color: #111111;
            }
        """)

        layout = QVBoxLayout(self)
        self._figure = Figure(figsize=(7.0, 4.6), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas)

    def clear_plot(self) -> None:
        self._figure.clear()
        self._figure.patch.set_facecolor("black")
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("black")
        ax.set_title("Contrast flow curves", color="white", fontsize=11)
        ax.set_xlabel("Frame index", color="white", fontsize=9)
        ax.set_ylabel("Mean intensity", color="white", fontsize=9)
        ax.tick_params(axis="both", colors="white", labelsize=8)

        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")
            spine.set_linewidth(0.8)

        ax.grid(True, color="#444444", linestyle="--", linewidth=0.5, alpha=0.5)
        self._figure.tight_layout()
        self._canvas.draw()

    def plot_multiple(self, curves_data: list[dict]) -> None:
        """
        Draw multiple stored curves.

        Each item in curves_data must contain:
            {
                "curve": np.ndarray,
                "cx": int,
                "cy": int,
                "radius": int,
                "color": str,   # matplotlib color, e.g. '#e6194b'
            }
        """
        self._figure.clear()
        self._figure.patch.set_facecolor("black")

        ax = self._figure.add_subplot(111)
        ax.set_facecolor("black")

        if not curves_data:
            ax.set_title("Contrast flow curves", color="white", fontsize=11)
            ax.set_xlabel("Frame index", color="white", fontsize=9)
            ax.set_ylabel("Mean intensity", color="white", fontsize=9)
        else:
            for item in curves_data:
                curve = item["curve"]
                cx = item["cx"]
                cy = item["cy"]
                radius = item["radius"]
                color = item["color"]

                frames = np.arange(len(curve))
                label = f"({cx}, {cy})  r={radius}"
                ax.plot(
                    frames,
                    curve,
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="round",
                    label=label,
                )

                peak_frame = int(np.argmax(curve))
                peak_val = float(curve[peak_frame])

                ax.scatter(
                    [peak_frame],
                    [peak_val],
                    color=color,
                    s=32,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.5,
                )

            ax.set_title(
                f"Contrast flow curves — {len(curves_data)} selected point(s)",
                color="white",
                fontsize=11,
                pad=10,
            )
            ax.set_xlabel("Frame index", color="white", fontsize=9)
            ax.set_ylabel("Mean intensity", color="white", fontsize=9)

        ax.tick_params(axis="both", colors="white", labelsize=8)

        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")
            spine.set_linewidth(0.8)

        ax.grid(True, color="#444444", linestyle="--", linewidth=0.5, alpha=0.5)

        if curves_data:
            legend = ax.legend(
                fontsize=8,
                facecolor="black",
                edgecolor="#888888",
                labelcolor="white",
                loc="upper right",
            )
            for text in legend.get_texts():
                text.set_color("white")

        self._figure.tight_layout()
        self._canvas.draw()


class Task3Widget(QWidget):
    """Task 3: DSA colour-coded visualisation with interactive flow curves."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._stack:       np.ndarray | None = None  # (N, H, W) float32
        self._time_map:    np.ndarray | None = None  # (H, W) float32
        self._bg_mask:     np.ndarray | None = None  # (H, W) bool
        self._lut:         vtk.vtkLookupTable | None = None
        self._image_actor: vtk.vtkImageActor  | None = None
        self._renderer:    vtk.vtkRenderer    | None = None
        self._iren:        vtk.vtkRenderWindowInteractor | None = None
        self._picking_enabled = True

        self._series_dirs: list[str] = []

        # Separate dialog for the flow-curve plot
        self._curve_dialog: FlowCurveDialog | None = None
        self._selected_curves = []
        self._next_curve_color = 0   # index in CHART_COLORS

        self._build_ui()

    # =========================================================================
    # UI
    # =========================================================================

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Left: VTK window ─────────────────────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(left)
        self.vtk_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        ll.addWidget(self.vtk_widget)
        splitter.addWidget(left)

        # ── Right: controls ───────────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(320)
        rl = QVBoxLayout(right)
        splitter.addWidget(right)

        # Series selection
        series_box = QGroupBox("DSA Series")
        sl = QVBoxLayout(series_box)
        sl.addWidget(QLabel("Available series:"))
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sl.addWidget(self._combo)

        btn_browse = QPushButton("Browse folder…")
        btn_browse.clicked.connect(self._browse_folder)
        sl.addWidget(btn_browse)

        btn_load = QPushButton("Load selected series")
        btn_load.clicked.connect(self._load_selected)
        sl.addWidget(btn_load)

        self._lbl_status = QLabel("No series loaded.")
        self._lbl_status.setWordWrap(True)
        sl.addWidget(self._lbl_status)
        rl.addWidget(series_box)

        # Coloring method
        method_box = QGroupBox("Coloring method")
        ml = QVBoxLayout(method_box)

        self._btn_argmax   = QRadioButton("Argmax  (frame of peak intensity)")
        self._btn_weighted = QRadioButton("Weighted avg  (Σ f·I_f / Σ I_f)")
        self._btn_argmax.setChecked(True)

        self._method_group = QButtonGroup(self)
        self._method_group.addButton(self._btn_argmax,   0)
        self._method_group.addButton(self._btn_weighted, 1)
        self._method_group.buttonClicked.connect(self._recompute_map)
        self._method_group.buttonClicked.connect(
            lambda _: self._update_method_desc()
        )

        ml.addWidget(self._btn_argmax)
        ml.addWidget(self._btn_weighted)

        self._lbl_method_desc = QLabel("")
        self._lbl_method_desc.setWordWrap(True)
        self._lbl_method_desc.setStyleSheet("color: gray; font-size: 10px;")
        ml.addWidget(self._lbl_method_desc)
        self._update_method_desc()
        rl.addWidget(method_box)

        # Neighbourhood radius
        nb_box = QGroupBox("Flow curve neighbourhood")
        nb_l = QVBoxLayout(nb_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Radius:"))
        self._slider_radius = QSlider(Qt.Horizontal)
        self._slider_radius.setRange(0, 15)
        self._slider_radius.setValue(DEFAULT_NEIGHBOUR_RADIUS)
        self._slider_radius.valueChanged.connect(
            lambda v: self._lbl_radius.setText(f"{v} px")
        )
        row.addWidget(self._slider_radius)
        self._lbl_radius = QLabel(f"{DEFAULT_NEIGHBOUR_RADIUS} px")
        self._lbl_radius.setFixedWidth(34)
        row.addWidget(self._lbl_radius)
        nb_l.addLayout(row)
        nb_l.addWidget(QLabel(
            "Left-click the DSA image to open the flow curve in a separate window."
        ))
        self._lbl_picked = QLabel("")
        nb_l.addWidget(self._lbl_picked)
        rl.addWidget(nb_box)

        btn_clear_curves = QPushButton("Clear selected curves")
        btn_clear_curves.clicked.connect(self._clear_selected_curves)
        nb_l.addWidget(btn_clear_curves)

        rl.addStretch()
        rl.addWidget(QLabel("Red = early frame\nBlue = late frame"))

    def _update_method_desc(self):
        """Update the descriptive text below the coloring method options."""
        mid = self._method_group.checkedId()
        descs = {
            0: ("Sharpest colour boundaries. Encodes only the\n"
                "single frame of maximum contrast."),
            1: ("Smoother result. Colour reflects the centre of\n"
                "mass of the contrast interval, not just the peak."),
        }
        self._lbl_method_desc.setText(descs.get(mid, ""))
    
    def _clear_selected_curves(self) -> None:
        """Clear the list of selected curves and close the curve dialog."""
        self._selected_curves.clear()
        self._next_curve_color = 0

        if self._curve_dialog is not None:
            self._curve_dialog.clear_plot()

    # =========================================================================
    # VTK initialisation
    # =========================================================================

    def _init_vtk(self):
        self._renderer = make_renderer(background=(0.0, 0.0, 0.0))
        rw = self.vtk_widget.GetRenderWindow()
        rw.AddRenderer(self._renderer)

        self._iren = rw.GetInteractor()

        # vtkInteractorStyleImage enables pan/zoom.
        self._style = vtk.vtkInteractorStyleImage()
        self._iren.SetInteractorStyle(self._style)

        self.vtk_widget.Initialize()

        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.005)
        self._iren.SetPicker(self._picker)

        # Priority 1.0 ensures this observer fires before the style's built-in
        # handlers, so we can pick first and then forward to the style.
        self._iren.AddObserver("LeftButtonPressEvent", self._on_left_click, 1.0)

    # =========================================================================
    # Series discovery
    # =========================================================================

    def _discover_series(self):
        """
        Scan the DSA directory for subfolders containing PNG series and
        populate the combo box. The user can also add more folders with
        the Browse button.
        """
        dsa_dir = PATHS.get("dsa_dir", "")
        if not os.path.isdir(dsa_dir):
            self._lbl_status.setText(f"DSA directory not found:\n{dsa_dir}")
            return

        self._combo.clear()
        self._series_dirs.clear()

        for entry in sorted(os.scandir(dsa_dir), key=lambda e: e.name):
            if entry.is_dir():
                if glob.glob(os.path.join(entry.path, "*.png")):
                    self._combo.addItem(entry.name)
                    self._series_dirs.append(entry.path)

        status = (
            f"Found {len(self._series_dirs)} series."
            if self._series_dirs else "No PNG series found."
        )
        self._lbl_status.setText(status)

    def _browse_folder(self):
        """
        Allow the user to select an additional folder containing a PNG series.
        If valid, add it to the combo box and select it.
        """
        folder = QFileDialog.getExistingDirectory(
            self, "Select PNG series folder", PATHS.get("dsa_dir", "")
        )
        if not folder:
            return
        if not glob.glob(os.path.join(folder, "*.png")):
            self._lbl_status.setText("No PNG files in that folder.")
            return
        if folder not in self._series_dirs:
            self._combo.addItem(os.path.basename(folder))
            self._series_dirs.append(folder)
        self._combo.setCurrentIndex(self._series_dirs.index(folder))

    def _load_selected(self):
        idx = self._combo.currentIndex()
        if 0 <= idx < len(self._series_dirs):
            self._load_series(self._series_dirs[idx])

    # =========================================================================
    # Core pipeline
    # =========================================================================

    def _load_series(self, folder: str):
        """
        Read every PNG in *folder* with vtkPNGReader, convert each frame to
        a greyscale NumPy array, and stack them into self._stack (N, H, W).

        Frames whose dimensions differ from the majority are discarded with
        a warning so that series with inconsistent sizes (e.g. Embo) still
        load correctly.
        """
        png_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not png_paths:
            self._lbl_status.setText("No PNGs found.")
            return

        self._lbl_status.setText(f"Loading {len(png_paths)} frames…")

        raw_frames: list[tuple[tuple, np.ndarray]] = []

        for path in png_paths:
            reader = vtk.vtkPNGReader()
            reader.SetFileName(path)
            reader.Update()

            img = reader.GetOutput()
            W   = img.GetDimensions()[0]
            H   = img.GetDimensions()[1]
            n_c = img.GetNumberOfScalarComponents()

            arr = vtk_numpy.vtk_to_numpy(
                img.GetPointData().GetScalars()
            )

            if n_c > 1:
                arr = arr.reshape(H, W, n_c)
                # Convert to luminance
                arr = (
                    0.299 * arr[:, :, 0]
                    + 0.587 * arr[:, :, 1]
                    + 0.114 * arr[:, :, 2]
                ).astype(np.float32)
            else:
                arr = arr.reshape(H, W).astype(np.float32)

            raw_frames.append(((H, W), arr))

        # Determine the most common frame size and discard outliers.
        # This handles series like Embo where a few frames have different dims.
        from collections import Counter
        size_counts = Counter(shape for shape, _ in raw_frames)
        dominant_size = size_counts.most_common(1)[0][0]
        n_total = len(raw_frames)

        frames = [arr for shape, arr in raw_frames if shape == dominant_size]
        n_discarded = n_total - len(frames)

        if not frames:
            self._lbl_status.setText("No consistent frames found.")
            return

        self._stack = np.stack(frames, axis=0)   # (N, H, W) float32
        N, H, W = self._stack.shape

        status = f"Loaded {N} frames · {W}×{H} px"
        if n_discarded:
            status += f"\n({n_discarded} frame(s) discarded — size mismatch)"
        status += f"\n{os.path.basename(folder)}"
        self._lbl_status.setText(status)

        self._selected_curves.clear()
        self._next_curve_color = 0

        if self._curve_dialog is not None:
            self._curve_dialog.clear_plot()

        self._recompute_map()

    def _recompute_map(self, _=None):
        """
        Compute self._time_map from self._stack using the selected method,
        then refresh the VTK display.
        """
        if self._stack is None:
            return

        N, H, W = self._stack.shape
        max_vals  = self._stack.max(axis=0)           # (H, W)
        threshold = max_vals.max() * BACKGROUND_THRESHOLD_PCT
        self._bg_mask = max_vals < threshold

        mid = self._method_group.checkedId()

        if mid == 0:
            # ── Argmax ───────────────────────────────────────────────────────
            # time_map(x,y) = argmax_f  I_f(x,y)
            time_map = np.argmax(self._stack, axis=0).astype(np.float32)

        else:
            # ── Intensity-weighted average ────────────────────────────────────
            # time_map(x,y) = Σ_f [f · I_f] / Σ_f [I_f]
            # Only frames above the background threshold contribute as weights
            # so that low-noise pixels don't bias the result toward the middle
            # of the temporal range.

            f_idx   = np.arange(N, dtype=np.float32)
            weights = np.where(self._stack > threshold, self._stack, 0.0)
            w_sum   = weights.sum(axis=0)

            time_map = np.where(
                w_sum > 0,
                (weights * f_idx[:, None, None]).sum(axis=0) / w_sum,
                0.0,
            ).astype(np.float32)

        time_map[self._bg_mask] = 0.0
        self._time_map = time_map
        self._push_to_vtk()

    def _push_to_vtk(self):
        """
        Convert self._time_map to a colour image and display it.
        """
        if self._time_map is None:
            return

        if not self._renderer:
            self._init_vtk()

        N = self._stack.shape[0]
        H, W = self._time_map.shape

        # numpy_to_vtk_image (utils.py) wraps a 2-D array as vtkImageData
        vtk_img = numpy_to_vtk_image(self._time_map)

        # Map the LUT range to the actual data range so the full colour spectrum
        # is always used, regardless of which coloring method is active.
        # For argmax the range is [0, N-1].
        # For weighted average the values cluster in a subrange,
        # so we stretch the LUT to [actual_min, actual_max] instead of [0, N-1].
        valid_vals = self._time_map[~self._bg_mask]
        if valid_vals.size > 0:
            lut_min = float(valid_vals.min())
            lut_max = float(valid_vals.max())
        else:
            lut_min, lut_max = 0.0, float(N - 1)

        self._lut = make_time_lut(N)
        self._lut.SetRange(lut_min, lut_max)
        self._lut.Build()

        # vtkImageMapToColors applies the LUT to every scalar value
        coloriser = vtk.vtkImageMapToColors()
        coloriser.SetInputData(vtk_img)
        coloriser.SetLookupTable(self._lut)
        coloriser.SetOutputFormatToRGB()
        coloriser.Update()

        colored = coloriser.GetOutput()
        self._mask_background(colored, H, W)

        self._renderer.RemoveAllViewProps()

        self._image_actor = vtk.vtkImageActor()
        self._image_actor.PickableOn()
        self._image_actor.GetMapper().SetInputData(colored)
        self._renderer.AddActor(self._image_actor)

        # vtkScalarBarActor provides the colour legend
        sbar = vtk.vtkScalarBarActor()
        sbar.SetLookupTable(self._lut)
        sbar.SetTitle("Frame")
        sbar.SetNumberOfLabels(5)
        sbar.SetPosition(0.88, 0.08)
        sbar.SetWidth(0.09)
        sbar.SetHeight(0.82)
        sbar.GetTitleTextProperty().SetFontSize(11)
        sbar.GetLabelTextProperty().SetFontSize(9)
        self._renderer.AddActor2D(sbar)

        self._renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _mask_background(
        self, vtk_colored: vtk.vtkImageData, H: int, W: int
    ) -> None:
        """Zero-out background pixels in the coloured VTK image."""
        if self._bg_mask is None: # bg_mask is True for background pixels
            return
        scalars = vtk_numpy.vtk_to_numpy(
            vtk_colored.GetPointData().GetScalars()
        )
        # bg_mask is in NumPy order; flip to match VTK row ordering
        mask_flat = np.flipud(self._bg_mask).ravel()
        scalars[mask_flat] = 0
        vtk_colored.GetPointData().GetScalars().Modified()

    # =========================================================================
    # Pixel picking and flow curve
    # =========================================================================

    def _on_left_click(self, _obj, _event) -> None:
        """
        Observer on LeftButtonPressEvent (priority 1.0, fires before the style).
        """
        # Get coordinates from the interactor
        x_scr, y_scr = self._iren.GetEventPosition()

        if self._stack is not None and self._image_actor is not None:
            self._picker.Pick(x_scr, y_scr, 0, self._renderer)

        if self._picker.GetCellId() >= 0:
            world = self._picker.GetPickPosition()
            N, H, W = self._stack.shape

            px     = int(np.clip(int(round(world[0])), 0, W - 1))
            py_vtk = int(np.clip(int(round(world[1])), 0, H - 1))
            py_np  = H - 1 - py_vtk

            if 0 <= px < W and 0 <= py_np < H:
                self._lbl_picked.setText(f"Picked: ({px}, {py_np})")
                self._show_flow_curve(px, py_np)

        # Always forward to the style so pan/zoom keeps working
        self._style.OnLeftButtonDown()

    def _show_flow_curve(self, cx: int, cy: int) -> None:
        """
        Given a picked pixel coordinate (cx, cy), extract the contrast flow curve
        at that location averaged over a square neighbourhood of radius r, then
        add it to the list of selected curves and display in the curve dialog.
        """
        N, H, W = self._stack.shape
        r = self._slider_radius.value()

        y0 = max(0, cy - r)
        y1 = min(H - 1, cy + r)
        x0 = max(0, cx - r)
        x1 = min(W - 1, cx + r)

        patch = self._stack[:, y0:y1 + 1, x0:x1 + 1]
        if patch.size == 0:
            return

        curve = patch.mean(axis=(1, 2))

        color = CHART_COLORS[self._next_curve_color % len(CHART_COLORS)]
        self._next_curve_color += 1

        self._selected_curves.append({
            "curve": curve,
            "cx": cx,
            "cy": cy,
            "radius": r,
            "color": color,
        })

        if self._curve_dialog is None:
            self._curve_dialog = FlowCurveDialog(self)

        self._curve_dialog.plot_multiple(self._selected_curves)
        self._curve_dialog.show()
        self._curve_dialog.raise_()
        self._curve_dialog.activateWindow()

    # =========================================================================
    # Qt lifecycle
    # =========================================================================

    def showEvent(self, event: object) -> None:
        super().showEvent(event)
        if self._renderer is None:
            self._init_vtk()
        if self._combo.count() == 0:
            self._discover_series()
            if self._series_dirs:
                self._load_series(self._series_dirs[0])