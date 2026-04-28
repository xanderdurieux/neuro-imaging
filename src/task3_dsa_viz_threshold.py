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
        self.setStyleSheet("QDialog { background-color: #111111; }")

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
                curve  = item["curve"]
                cx     = item["cx"]
                cy     = item["cy"]
                radius = item["radius"]
                color  = item["color"]

                frames = np.arange(len(curve))
                ax.plot(frames, curve, color=color, linewidth=2.0,
                        solid_capstyle="round", label=f"({cx}, {cy})  r={radius}")

                peak_frame = int(np.argmax(curve))
                ax.scatter([peak_frame], [curve[peak_frame]], color=color,
                           s=32, zorder=5, edgecolors="white", linewidths=0.5)

            ax.set_title(
                f"Contrast flow curves — {len(curves_data)} selected point(s)",
                color="white", fontsize=11, pad=10)
            ax.set_xlabel("Frame index", color="white", fontsize=9)
            ax.set_ylabel("Mean intensity", color="white", fontsize=9)

            legend = ax.legend(fontsize=8, facecolor="black",
                               edgecolor="#888888", labelcolor="white",
                               loc="upper right")
            for text in legend.get_texts():
                text.set_color("white")

        ax.tick_params(axis="both", colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#bbbbbb")
            spine.set_linewidth(0.8)
        ax.grid(True, color="#444444", linestyle="--", linewidth=0.5, alpha=0.5)
        self._figure.tight_layout()
        self._canvas.draw()


class Task3Widget(QWidget):
    """Task 3: DSA colour-coded visualisation with interactive flow curves."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # All raw frames of the loaded folder, as a list of (H,W) arrays.
        # Used to build _views and to drive the frame slider
        self._all_frames:  list[np.ndarray] = []

        # List of views: each view is a numpy stack (N, H, W) of the frames
        # that belong to that view, with the anomalous last frame removed
        self._views:       list[np.ndarray] = []

        # The view currently selected by the user (index into _views)
        self._current_view_idx: int = 0

        # The stack of the current view (this is what _recompute_map works on)
        self._stack:       np.ndarray | None = None  # (N, H, W) float32

        # The 2-D time map produced by _recompute_map (H, W) float32
        self._time_map:    np.ndarray | None = None

        # VTK objects
        self._lut:         vtk.vtkLookupTable | None = None
        self._image_actor: vtk.vtkImageActor  | None = None
        self._renderer:    vtk.vtkRenderer    | None = None
        self._iren:        vtk.vtkRenderWindowInteractor | None = None

        # Whether we are currently showing the DSA map (True) or a raw frame (False)
        self._showing_dsa: bool = False

        self._series_dirs: list[str] = []

        # Flow curve dialog
        self._curve_dialog: FlowCurveDialog | None = None
        self._selected_curves = []
        self._next_curve_color = 0

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
        self.vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ll.addWidget(self.vtk_widget)
        splitter.addWidget(left)

        # ── Right: controls ───────────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(320)
        rl = QVBoxLayout(right)
        splitter.addWidget(right)

        # ── Series selection ──────────────────────────────────────────────────
        series_box = QGroupBox("DSA Series")
        sl = QVBoxLayout(series_box)

        sl.addWidget(QLabel("Available series:"))
        self._combo_series = QComboBox()
        self._combo_series.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sl.addWidget(self._combo_series)

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

        # ── View selection ────────────────────────────────────────────────────
        view_box = QGroupBox("View")
        vl = QVBoxLayout(view_box)

        # Each folder may contain 2 views concatenated.
        # _split_into_views detects the boundary and exposes them here.
        self._combo_view = QComboBox()
        self._combo_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo_view.currentIndexChanged.connect(self._on_view_changed)
        vl.addWidget(self._combo_view)
        rl.addWidget(view_box)

        # ── Frame slider ──────────────────────────────────────────────────────
        frame_box = QGroupBox("Frame browser")
        fl = QVBoxLayout(frame_box)

        row_f = QHBoxLayout()
        row_f.addWidget(QLabel("Frame:"))
        self._slider_frame = QSlider(Qt.Horizontal)
        self._slider_frame.setRange(0, 0)
        self._slider_frame.setValue(0)
        self._slider_frame.valueChanged.connect(self._on_frame_slider)
        row_f.addWidget(self._slider_frame)
        self._lbl_frame = QLabel("0 / 0")
        self._lbl_frame.setFixedWidth(50)
        row_f.addWidget(self._lbl_frame)
        fl.addLayout(row_f)

        fl.addWidget(QLabel("Drag to browse raw frames of this view."))
        rl.addWidget(frame_box)

        # ── DSA button ────────────────────────────────────────────────────────
        self._btn_dsa = QPushButton("Show DSA map")
        self._btn_dsa.setToolTip(
            "Compute and display the colour-coded time map for the current view."
        )
        self._btn_dsa.clicked.connect(self._show_dsa)
        rl.addWidget(self._btn_dsa)

        # ── Coloring method ───────────────────────────────────────────────────
        method_box = QGroupBox("Coloring method")
        ml = QVBoxLayout(method_box)

        self._btn_argmax   = QRadioButton("Argmax  (frame of peak contrast)")
        self._btn_weighted = QRadioButton("Weighted avg  (Σ f·w_f / Σ w_f)")
        self._btn_argmax.setChecked(True)

        self._method_group = QButtonGroup(self)
        self._method_group.addButton(self._btn_argmax,   0)
        self._method_group.addButton(self._btn_weighted, 1)
        # Changing method only recomputes if we are already showing the DSA
        self._method_group.buttonClicked.connect(self._on_method_changed)

        ml.addWidget(self._btn_argmax)
        ml.addWidget(self._btn_weighted)

        self._lbl_method_desc = QLabel("")
        self._lbl_method_desc.setWordWrap(True)
        self._lbl_method_desc.setStyleSheet("color: gray; font-size: 10px;")
        ml.addWidget(self._lbl_method_desc)
        self._update_method_desc()
        rl.addWidget(method_box)

        # ── Flow curve neighbourhood ──────────────────────────────────────────
        nb_box = QGroupBox("Flow curve neighbourhood")
        nb_l = QVBoxLayout(nb_box)

        row_r = QHBoxLayout()
        row_r.addWidget(QLabel("Radius:"))
        self._slider_radius = QSlider(Qt.Horizontal)
        self._slider_radius.setRange(0, 15)
        self._slider_radius.setValue(DEFAULT_NEIGHBOUR_RADIUS)
        self._slider_radius.valueChanged.connect(
            lambda v: self._lbl_radius.setText(f"{v} px")
        )
        row_r.addWidget(self._slider_radius)
        self._lbl_radius = QLabel(f"{DEFAULT_NEIGHBOUR_RADIUS} px")
        self._lbl_radius.setFixedWidth(34)
        row_r.addWidget(self._lbl_radius)
        nb_l.addLayout(row_r)

        nb_l.addWidget(QLabel(
            "Left-click the DSA map to plot the flow curve at that pixel."
        ))
        self._lbl_picked = QLabel("")
        nb_l.addWidget(self._lbl_picked)

        btn_clear = QPushButton("Clear selected curves")
        btn_clear.clicked.connect(self._clear_selected_curves)
        nb_l.addWidget(btn_clear)

        rl.addWidget(nb_box)

        rl.addStretch()
        rl.addWidget(QLabel("Red = early frame    Blue = late frame"))

    def _update_method_desc(self):
        mid = self._method_group.checkedId()
        descs = {
            0: "Sharpest boundaries. Uses only the frame of maximum contrast.",
            1: "Smoother. Colour = intensity-weighted mean of frame indices.",
        }
        self._lbl_method_desc.setText(descs.get(mid, ""))

    def _on_method_changed(self, _):
        self._update_method_desc()
        # Only recompute if the DSA map is currently visible
        if self._showing_dsa:
            self._show_dsa()

    def _clear_selected_curves(self) -> None:
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
        self._style = vtk.vtkInteractorStyleImage()
        self._iren.SetInteractorStyle(self._style)
        self.vtk_widget.Initialize()

        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.005)
        self._iren.SetPicker(self._picker)

        self._iren.AddObserver("LeftButtonPressEvent", self._on_left_click, 1.0)

    # =========================================================================
    # Series discovery
    # =========================================================================

    def _discover_series(self):
        dsa_dir = PATHS.get("dsa_dir", "")
        if not os.path.isdir(dsa_dir):
            self._lbl_status.setText(f"DSA directory not found:\n{dsa_dir}")
            return

        self._combo_series.clear()
        self._series_dirs.clear()

        for entry in sorted(os.scandir(dsa_dir), key=lambda e: e.name):
            if entry.is_dir() and glob.glob(os.path.join(entry.path, "*.png")):
                self._combo_series.addItem(entry.name)
                self._series_dirs.append(entry.path)

        self._lbl_status.setText(
            f"Found {len(self._series_dirs)} series."
            if self._series_dirs else "No PNG series found."
        )

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select PNG series folder", PATHS.get("dsa_dir", "")
        )
        if not folder:
            return
        if not glob.glob(os.path.join(folder, "*.png")):
            self._lbl_status.setText("No PNG files in that folder.")
            return
        if folder not in self._series_dirs:
            self._combo_series.addItem(os.path.basename(folder))
            self._series_dirs.append(folder)
        self._combo_series.setCurrentIndex(self._series_dirs.index(folder))

    def _load_selected(self):
        idx = self._combo_series.currentIndex()
        if 0 <= idx < len(self._series_dirs):
            self._load_series(self._series_dirs[idx])

    # =========================================================================
    # Loading & view splitting
    # =========================================================================

    def _load_series(self, folder: str):
        """
        Read all PNGs in folder, keep only those with the dominant resolution,
        detect view boundaries, and populate the view combo box.
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

            arr = vtk_numpy.vtk_to_numpy(img.GetPointData().GetScalars())

            arr = arr.reshape(H, W).astype(np.float32)

            raw_frames.append(((H, W), arr))

        # Discard frames whose resolution differs from the majority
        from collections import Counter
        dominant_size = Counter(s for s, _ in raw_frames).most_common(1)[0][0]
        frames = [arr for s, arr in raw_frames if s == dominant_size]
        n_discarded = len(raw_frames) - len(frames)

        if not frames:
            self._lbl_status.setText("No consistent frames found.")
            return

        self._all_frames = frames

        # Split into views and populate the view combo
        self._views = self._split_into_views(frames)

        self._combo_view.blockSignals(True)
        self._combo_view.clear()
        for i, v in enumerate(self._views):
            self._combo_view.addItem(f"View {i + 1}  ({v.shape[0]} frames)")
        self._combo_view.blockSignals(False)

        # Select view 1 and show its first frame
        self._select_view(0)

        n_total = len(frames)
        status = (f"{len(self._views)} view(s) detected · "
                  f"{n_total} usable frames · {dominant_size[1]}×{dominant_size[0]} px"
                  f"\n{os.path.basename(folder)}")
        if n_discarded:
            status += f"\n({n_discarded} frame(s) discarded — size mismatch)"
        self._lbl_status.setText(status)

        # Reset curves
        self._selected_curves.clear()
        self._next_curve_color = 0
        if self._curve_dialog is not None:
            self._curve_dialog.clear_plot()

    def _split_into_views(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Detect view boundaries by looking for a sudden jump in mean pixel
        intensity between consecutive frames.

        The anomalous frame that causes the jump is the LAST frame of its
        view (e.g. a bright reference image), so we exclude it and start the
        next view with the frame that follows it.

        Returns a list of numpy stacks (N, H, W), one per view.
        """
        if len(frames) <= 1:
            return [np.stack(frames, axis=0)]

        means = np.array([f.mean() for f in frames])
        MIN_JUMP = 10.0
        anomalous = set()
        for i in range(1, len(means) - 1):
            left_diff  = means[i - 1] - means[i]
            right_diff = means[i + 1] - means[i]   
            
            if left_diff > MIN_JUMP and right_diff > MIN_JUMP:
                anomalous.add(i)

        if len(means) >= 2 and (means[-2] - means[-1]) > MIN_JUMP:
            anomalous.add(len(means) - 1)

        print(f"Anomalous frames detected: {sorted(anomalous)}")

        views = []
        current_start = 0

        for i in range(len(frames)):
            if i in anomalous:
                segment = [f for j, f in enumerate(frames[current_start:i])
                        if (current_start + j) not in anomalous]
                if len(segment) >= 3:
                    views.append(np.stack(segment, axis=0))
                current_start = i + 1

        # Add the final segment (no anomalous frame at its end)
        segment = [f for j, f in enumerate(frames[current_start:])
               if (current_start + j) not in anomalous]
        if len(segment) >= 3:
            views.append(np.stack(segment, axis=0))
            
        print(f"Views found: {len(views)}, sizes: {[v.shape[0] for v in views]}")
        return views if views else [np.stack(frames, axis=0)]

    # =========================================================================
    # View & frame selection
    # =========================================================================

    def _select_view(self, view_idx: int) -> None:
        """
        Switch to a different view: update _stack, reset the frame slider,
        and show the first raw frame.
        """
        if not self._views or view_idx >= len(self._views):
            return

        self._current_view_idx = view_idx
        self._stack = self._views[view_idx]   # (N, H, W)
        N = self._stack.shape[0]

        # Update slider range for this view
        self._slider_frame.blockSignals(True)
        self._slider_frame.setRange(0, N - 1)
        self._slider_frame.setValue(0)
        self._slider_frame.blockSignals(False)
        self._lbl_frame.setText(f"0 / {N - 1}")

        # Show the first raw frame and exit DSA mode
        self._showing_dsa = False
        self._show_raw_frame(0)

    def _on_view_changed(self, idx: int) -> None:
        """Called when the user picks a different view in the combo box."""
        self._select_view(idx)

    def _on_frame_slider(self, value: int) -> None:
        """Called when the user drags the frame slider."""
        if self._stack is None:
            return
        N = self._stack.shape[0]
        self._lbl_frame.setText(f"{value} / {N - 1}")
        # Switching to raw-frame mode
        self._showing_dsa = False
        self._show_raw_frame(value)

    # =========================================================================
    # Display helpers
    # =========================================================================

    def _show_raw_frame(self, frame_idx: int) -> None:
        """
        Display a single raw greyscale frame in the VTK window.
        The frame is taken from self._stack (current view).
        """
        if self._stack is None:
            return
        if not self._renderer:
            self._init_vtk()

        frame = self._stack[frame_idx]   # (H, W) float32

        # Wrap the greyscale array as a VTK image and display it directly
        vtk_img = numpy_to_vtk_image(frame)

        # Map the grey values through a greyscale LUT so VTK renders them as RGB
        grey_lut = vtk.vtkLookupTable()
        grey_lut.SetHueRange(0.0, 0.0)
        grey_lut.SetSaturationRange(0.0, 0.0)
        grey_lut.SetValueRange(0.0, 1.0)
        grey_lut.SetRange(float(frame.min()), float(frame.max()))
        grey_lut.SetNumberOfColors(256)
        grey_lut.Build()

        coloriser = vtk.vtkImageMapToColors()
        coloriser.SetInputData(vtk_img)
        coloriser.SetLookupTable(grey_lut)
        coloriser.SetOutputFormatToRGB()
        coloriser.Update()

        self._renderer.RemoveAllViewProps()

        self._image_actor = vtk.vtkImageActor()
        self._image_actor.PickableOn()
        self._image_actor.GetMapper().SetInputData(coloriser.GetOutput())
        self._renderer.AddActor(self._image_actor)

        self._renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _show_dsa(self) -> None:
        """
        Compute the DSA time map for the current view and display it
        with the red -> blue colour scale.
        """
        if self._stack is None:
            return
        self._showing_dsa = True
        self._recompute_map()

    def _recompute_map(self) -> None:
        """
        Compute self._time_map from self._stack (current view only).

        For each frame, subtract the median gray value of that frame from
        every pixel, and take the absolute value.
        """
        if self._stack is None:
            return

        N, H, W = self._stack.shape
        mid = self._method_group.checkedId()

        # The median is an estimator of the background intensity
        median_val = np.median(self._stack)
        stack_adj = np.abs(self._stack - median_val) # (N, H, W)

        # Now the background is close to 0, and pixels that differ from it have larger values.
        strength = stack_adj.max(axis=0)

        threshold = 0.15 * strength.max()
        mask = strength > threshold

        if mid == 0:
            # Frame with maximum intensity
            raw_map = np.argmax(stack_adj, axis=0).astype(np.float32)
        else:
            # Intensity-weighted average of frame indices: Σ f·w_f / Σ w_f
            f_idx = np.arange(N, dtype=np.float32)
            w_sum = stack_adj.sum(axis=0) # (H, W) sum of weights for each pixel
            raw_map = np.where(
                w_sum > 0,
                (stack_adj * f_idx[:, None, None]).sum(axis=0) / w_sum,
                0.0
            ).astype(np.float32) # (H, W) weighted average of frame indices, or 0 if all weights are zero

        time_map = np.where(mask, raw_map, -1.0)
        self._time_map = time_map
        self._push_dsa_to_vtk()

    def _push_dsa_to_vtk(self) -> None:
        """
        Convert self._time_map to a VTK image and display it with the time LUT.
        Pixels with no signal (time < 0) are masked out and shown as dark gray.
        """
        
        if self._time_map is None:
            return
        if not self._renderer:
            self._init_vtk()

        N = self._stack.shape[0]

        display_map = self._time_map.copy()
        no_signal = display_map < 0 # Mask for pixels with no signal (time < 0)
        display_map[no_signal] = 0.0 # Set no-signal pixels to 0 for the Gaussian blur and color mapping steps

        vtk_img = numpy_to_vtk_image(display_map)
        
        gauss = vtk.vtkImageGaussianSmooth() # Apply a mild Gaussian blur to make the colour transitions smoother
        gauss.SetInputData(vtk_img) # The input is the raw time map (with no-signal pixels set to 0)
        gauss.SetStandardDeviations(0.7, 0.7, 0.0) # Sigma controls the amount of blur
        gauss.SetRadiusFactors(1.5, 1.5, 0.0) # Radius controls the extent of the blur
        gauss.Update()

        vtk_img = gauss.GetOutput()

        self._lut = make_time_lut(N)
        self._lut.SetRange(0, N - 1)
        self._lut.Build()

        coloriser = vtk.vtkImageMapToColors()
        coloriser.SetInputData(vtk_img)
        coloriser.SetLookupTable(self._lut)
        coloriser.SetOutputFormatToRGB()
        coloriser.Update()

        # Extract the RGB array from the coloriser output, reshape it to (H, W, 3), and set no-signal pixels to dark gray
        rgb = vtk_numpy.vtk_to_numpy(
            coloriser.GetOutput().GetPointData().GetScalars()
        ).reshape(self._time_map.shape[0], self._time_map.shape[1], 3).copy()

        

        rgb[no_signal] = [13, 13, 13]

        flat = rgb.reshape(-1, 3)
        vtk_arr = vtk_numpy.numpy_to_vtk(flat, deep=True)
        vtk_arr.SetNumberOfComponents(3)

        masked_img = vtk.vtkImageData()
        masked_img.SetDimensions(
            self._time_map.shape[1], self._time_map.shape[0], 1
        )
        masked_img.GetPointData().SetScalars(vtk_arr)

        self._renderer.RemoveAllViewProps()

        # Create an image actor for the masked RGB image and add it to the renderer
        self._image_actor = vtk.vtkImageActor()
        self._image_actor.PickableOn()
        self._image_actor.GetMapper().SetInputData(masked_img)
        self._renderer.AddActor(self._image_actor)

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

    # =========================================================================
    # Pixel picking and flow curve
    # =========================================================================

    def _on_left_click(self, _obj, _event) -> None:
        """Left-click picks a pixel and plots its contrast flow curve.
        Only active when the DSA map is displayed.
        """
        x_scr, y_scr = self._iren.GetEventPosition()

        if self._showing_dsa and self._stack is not None and self._image_actor is not None:
            self._picker.Pick(x_scr, y_scr, 0, self._renderer)

            if self._picker.GetCellId() >= 0:
                world = self._picker.GetPickPosition()
                N, H, W = self._stack.shape

                px     = int(np.clip(int(round(world[0])), 0, W - 1))
                py_np = int(np.clip(int(round(world[1])), 0, H - 1))

                if 0 <= px < W and 0 <= py_np < H:
                    picked_time = self._time_map[py_np, px]

                    if picked_time < 0:
                        self._lbl_picked.setText(
                            f"Picked: ({px}, {py_np})  no signal"
                        )
                        return

                    self._lbl_picked.setText(
                        f"Picked: ({px}, {py_np})  time={picked_time:.2f}"
                    )

                    self._show_flow_curve(px, py_np)

        self._style.OnLeftButtonDown()

    def _show_flow_curve(self, cx: int, cy: int) -> None:
        """Extract and display the contrast flow curve at (cx, cy)."""
        N, H, W = self._stack.shape
        r = self._slider_radius.value()

        y0 = max(0, cy - r)
        y1 = min(H - 1, cy + r)
        x0 = max(0, cx - r)
        x1 = min(W - 1, cx + r)

        # Use the same contrast signal used for the DSA map
        median_val = np.median(self._stack)
        stack_adj = np.abs(self._stack - median_val)

        patch = stack_adj[:, y0:y1 + 1, x0:x1 + 1]
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
        if self._combo_series.count() == 0:
            self._discover_series()
            if self._series_dirs:
                self._load_series(self._series_dirs[0])