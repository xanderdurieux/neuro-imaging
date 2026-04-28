"""
Task 2.2 – EEG Visualization
==============================
• User places up to 8 electrode positions on the brain mesh via vtkPointPicker.
  (Picker is restricted to the brain actor only.)
• Each electrode is assigned a random value in {0, 50, 100}.
• Mesh vertices are colored using Inverse-Distance Weighting (IDW):

    Squared (w_i = 1/d_i²):
        v(p) = Σ [v_i / d_i²] / Σ [1 / d_i²]

  In both cases vertices coinciding with an electrode (d < ε) are assigned that
  electrode's value directly.

• Electrode spheres are placed using vtkGlyph3D.
• A QTimer fires every 5 s, re-randomises electrode values, recolors the mesh.
• Signal history for each electrode is shown in a per-electrode subplot grid
  (window width T is configurable via spin box; only last T samples shown).
• Semi-transparent skin mesh overlaid for context.
"""

import random
import os
import vtk
import numpy as np
from vtkmodules.util import numpy_support as vtk_numpy
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QPushButton, QLabel, QFileDialog, QSplitter, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .utils import PATHS, make_renderer, add_axes, make_rainbow_lut

# ── Constants ─────────────────────────────────────────────────────────────────

N_ELECTRODES      = 8
ELECTRODE_VALUES  = [0, 50, 100]
TIMER_INTERVAL_MS = 5000    # 5 seconds
DEFAULT_T         = 20      # default signal-history window (samples)

# Iso-values derived from actual label analysis (see task1):
#   0=bg, 42=skin, 84=skull, 127=grey matter, 169=brain, 254=lesion
BRAIN_ISO = 148   # midpoint between 127 and 169  → brain (white matter) surface
SKIN_ISO  =  21   # midpoint between   0 and  42  → outer head surface

CHART_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]

# Fixed smoothing setup for Task 2 brain mesh (non-configurable).
BRAIN_SMOOTH_ITERATIONS = 150
BRAIN_SMOOTH_PASSBAND   = 0.0003


class Task2Widget(QWidget):
    """Main widget for Task 2: EEG visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._reader         = None
        self._brain_contour: vtk.vtkContourFilter | None = None
        self._brain_poly:    vtk.vtkPolyData | None = None
        self._locator:       vtk.vtkPointLocator | None = None
        self._electrode_positions: list[tuple[float, float, float]] = []
        self._electrode_values:    list[int]   = []
        self._signal_history:      list[list[int]] = [[] for _ in range(N_ELECTRODES)]
        self._picking_enabled = False

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Left: 3-D VTK window ──
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(left)
        ll.addWidget(self.vtk_widget)
        splitter.addWidget(left)

        # ── Right: controls + per-electrode signal charts ──
        right = QWidget()
        rl = QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setSizes([800, 500])

        # Load
        btn_load = QPushButton("Load head_with_lesion.vtk")
        btn_load.clicked.connect(self._load_file)
        rl.addWidget(btn_load)

        # Electrodes
        elec_box = QGroupBox("Electrodes")
        el = QVBoxLayout(elec_box)

        self.lbl_count = QLabel("Placed: 0 / 8")
        el.addWidget(self.lbl_count)

        self.btn_pick = QPushButton("Enable Picking")
        self.btn_pick.setCheckable(True)
        self.btn_pick.toggled.connect(self._toggle_picking)
        el.addWidget(self.btn_pick)

        btn_clear = QPushButton("Clear Electrodes")
        btn_clear.clicked.connect(self._clear_electrodes)
        el.addWidget(btn_clear)

        rl.addWidget(elec_box)

        # Timer
        timer_box = QGroupBox("EEG Simulation (5 s timer)")
        tl = QVBoxLayout(timer_box)

        self.btn_timer = QPushButton("Start Timer")
        self.btn_timer.setCheckable(True)
        self.btn_timer.toggled.connect(self._toggle_timer)
        tl.addWidget(self.btn_timer)

        self.lbl_timer = QLabel("Timer: stopped")
        tl.addWidget(self.lbl_timer)

        # Window length T
        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("Signal window T:"))
        self.spin_T = QSpinBox()
        self.spin_T.setRange(5, 200)
        self.spin_T.setValue(DEFAULT_T)
        self.spin_T.setSuffix(" samples")
        row_t.addWidget(self.spin_T)
        tl.addLayout(row_t)

        rl.addWidget(timer_box)

        # Per-electrode signal charts (2 rows × 4 cols)
        chart_box = QGroupBox("Electrode Signals  (one subplot per electrode)")
        cl = QVBoxLayout(chart_box)
        self.figure  = Figure(figsize=(5, 4), dpi=80)
        self.canvas  = FigureCanvas(self.figure)
        cl.addWidget(self.canvas)
        rl.addWidget(chart_box)

        # Qt timer
        self._qtimer = QTimer(self)
        self._qtimer.setInterval(TIMER_INTERVAL_MS)
        self._qtimer.timeout.connect(self._on_timer_tick)

    # ── VTK setup ─────────────────────────────────────────────────────────────

    def _init_vtk(self):
        self._renderer = make_renderer()
        rw = self.vtk_widget.GetRenderWindow()
        rw.AddRenderer(self._renderer)
        self._iren = rw.GetInteractor()

        style = vtk.vtkInteractorStyleTrackballCamera()
        self._iren.SetInteractorStyle(style)
        self._style = style

        # CellPicker: true ray-surface intersection → always hits front face
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        self._iren.SetPicker(self._picker)
        self._iren.AddObserver("LeftButtonPressEvent", self._on_left_click)

        self._locator: vtk.vtkPointLocator | None = None

        add_axes(self._renderer, self._iren)
        self.vtk_widget.Initialize()

        # Scalar bar (EEG color range 0-100, blue→red)
        self._lut = make_rainbow_lut(0, 100)
        sbar = vtk.vtkScalarBarActor()
        sbar.SetLookupTable(self._lut)
        sbar.SetTitle("EEG")
        sbar.SetNumberOfLabels(3)
        sbar.SetPosition(0.87, 0.1)
        sbar.SetWidth(0.10)
        sbar.SetHeight(0.8)
        self._renderer.AddActor2D(sbar)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _load_file(self):
        default = PATHS["head"] if os.path.isfile(PATHS["head"]) else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open head_with_lesion.vtk", default, "VTK files (*.vtk)"
        )
        if path:
            self._setup_pipeline(path)

    def _setup_pipeline(self, path: str):
        if not hasattr(self, "_renderer"):
            self._init_vtk()
        else:
            self._renderer.RemoveAllViewProps()
            # Restore scalar bar after clearing props
            self._init_vtk.__func__   # scalar bar already in renderer

        self._reader = vtk.vtkStructuredPointsReader()
        self._reader.SetFileName(path)
        self._reader.Update()

        # ── Brain mesh ──
        self._brain_contour = vtk.vtkContourFilter()
        self._brain_contour.SetInputConnection(self._reader.GetOutputPort())
        self._brain_contour.SetValue(0, BRAIN_ISO)
        self._brain_contour.Update()

        smooth_poly = self._run_brain_smooth()

        # Mutable copy for per-vertex EEG coloring
        self._brain_poly = vtk.vtkPolyData()
        self._brain_poly.DeepCopy(smooth_poly)
        self._init_mesh_scalars()

        self._brain_mapper = vtk.vtkPolyDataMapper()
        self._brain_mapper.SetInputData(self._brain_poly)
        self._brain_mapper.SetLookupTable(self._lut)
        self._brain_mapper.SetScalarRange(0, 100)
        self._brain_mapper.ScalarVisibilityOn()

        self._brain_actor = vtk.vtkActor()
        self._brain_actor.SetMapper(self._brain_mapper)
        self._renderer.AddActor(self._brain_actor)

        # Restrict CellPicker to brain actor only (ignores skin + glyph actors)
        self._picker.InitializePickList()
        self._picker.AddPickList(self._brain_actor)
        self._picker.PickFromListOn()

        # Build point locator for fast nearest-vertex snapping after a cell pick
        self._build_locator()

        # ── Skin mesh (semi-transparent white overlay) ──
        skin_contour = vtk.vtkContourFilter()
        skin_contour.SetInputConnection(self._reader.GetOutputPort())
        skin_contour.SetValue(0, SKIN_ISO)

        skin_smooth = vtk.vtkWindowedSincPolyDataFilter()
        skin_smooth.SetInputConnection(skin_contour.GetOutputPort())
        skin_smooth.SetNumberOfIterations(30)
        skin_smooth.SetPassBand(0.05)
        skin_smooth.BoundarySmoothingOn()
        skin_smooth.NormalizeCoordinatesOn()

        skin_normals = vtk.vtkPolyDataNormals()
        skin_normals.SetInputConnection(skin_smooth.GetOutputPort())

        skin_mapper = vtk.vtkPolyDataMapper()
        skin_mapper.SetInputConnection(skin_normals.GetOutputPort())
        skin_mapper.ScalarVisibilityOff()

        skin_actor = vtk.vtkActor()
        skin_actor.SetMapper(skin_mapper)
        skin_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        skin_actor.GetProperty().SetOpacity(0.15)
        self._renderer.AddActor(skin_actor)

        # ── Electrode glyph pipeline ──
        sphere_src = vtk.vtkSphereSource()
        sphere_src.SetRadius(2.5)
        sphere_src.SetThetaResolution(12)
        sphere_src.SetPhiResolution(12)

        self._electrode_points_vtk = vtk.vtkPoints()
        self._electrode_pd         = vtk.vtkPolyData()
        self._electrode_pd.SetPoints(self._electrode_points_vtk)

        self._glyph = vtk.vtkGlyph3D()
        self._glyph.SetSourceConnection(sphere_src.GetOutputPort())
        self._glyph.SetInputData(self._electrode_pd)
        self._glyph.ScalingOff()

        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(self._glyph.GetOutputPort())

        self._glyph_actor = vtk.vtkActor()
        self._glyph_actor.SetMapper(glyph_mapper)
        self._glyph_actor.GetProperty().SetColor(1.0, 1.0, 0.0)   # yellow
        self._glyph_actor.GetProperty().SetAmbient(0.5)
        self._renderer.AddActor(self._glyph_actor)

        # Reset state
        self._electrode_positions.clear()
        self._electrode_values.clear()
        self._signal_history = [[] for _ in range(N_ELECTRODES)]
        self.lbl_count.setText("Placed: 0 / 8")

        self._renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self._update_chart()

    # ── Smoothing helpers ─────────────────────────────────────────────────────

    def _run_brain_smooth(self) -> vtk.vtkPolyData:
        """Run vtkWindowedSincPolyDataFilter with fixed smoothing parameters."""

        sinc = vtk.vtkWindowedSincPolyDataFilter()
        sinc.SetInputConnection(self._brain_contour.GetOutputPort())
        sinc.SetNumberOfIterations(BRAIN_SMOOTH_ITERATIONS)
        sinc.SetPassBand(BRAIN_SMOOTH_PASSBAND)
        sinc.BoundarySmoothingOn()
        sinc.NonManifoldSmoothingOn()
        sinc.NormalizeCoordinatesOn()
        sinc.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(sinc.GetOutputPort())
        normals.SetFeatureAngle(60.0)
        normals.Update()

        return normals.GetOutput()

    # ── Scalar initialisation ─────────────────────────────────────────────────

    def _init_mesh_scalars(self):
        """Fill all vertices with neutral value 50 using numpy (fast)."""
        n_pts    = self._brain_poly.GetNumberOfPoints()
        init_arr = np.full(n_pts, 50.0, dtype=np.float32)
        vtk_arr  = vtk_numpy.numpy_to_vtk(init_arr, deep=True,
                                           array_type=vtk.VTK_FLOAT)
        vtk_arr.SetName("EEG")
        self._brain_poly.GetPointData().SetScalars(vtk_arr)

    def _build_locator(self):
        """Build a vtkPointLocator on the brain mesh for fast nearest-vertex lookup."""
        self._locator = vtk.vtkPointLocator()
        self._locator.SetDataSet(self._brain_poly)
        self._locator.BuildLocator()

    # ── Electrode picking ─────────────────────────────────────────────────────

    def _toggle_picking(self, enabled: bool):
        self._picking_enabled = enabled

    def _on_left_click(self, obj, event):
        if not self._picking_enabled:
            self._style.OnLeftButtonDown()
            return
        if len(self._electrode_positions) >= N_ELECTRODES or self._brain_poly is None:
            return

        x, y = self._iren.GetEventPosition()
        self._picker.Pick(x, y, 0, self._renderer)

        # vtkCellPicker: negative cell ID means the ray missed the brain mesh
        if self._picker.GetCellId() < 0:
            return

        # GetPickPosition() gives the exact 3-D point on the front-facing surface.
        # Snap it to the nearest mesh vertex so the electrode sits on the surface.
        hit_pos = self._picker.GetPickPosition()
        pid     = self._locator.FindClosestPoint(hit_pos)
        pos     = self._brain_poly.GetPoint(pid)

        val = random.choice(ELECTRODE_VALUES)

        self._electrode_positions.append(pos)
        self._electrode_values.append(val)
        idx = len(self._electrode_positions) - 1
        self._signal_history[idx].append(val)

        # Update glyph
        self._electrode_points_vtk.InsertNextPoint(*pos)
        self._electrode_points_vtk.Modified()
        self._electrode_pd.Modified()
        self._glyph.Update()

        self.lbl_count.setText(f"Placed: {len(self._electrode_positions)} / {N_ELECTRODES}")
        self._recolor_mesh()
        self._update_chart()
        self.vtk_widget.GetRenderWindow().Render()

    def _clear_electrodes(self):
        self._electrode_positions.clear()
        self._electrode_values.clear()
        self._signal_history = [[] for _ in range(N_ELECTRODES)]
        if self._brain_poly is not None:
            self._electrode_points_vtk.Reset()
            self._electrode_pd.Modified()
            self._glyph.Update()
            self.lbl_count.setText("Placed: 0 / 8")
            self._init_mesh_scalars()
            self._brain_poly.Modified()
            self.vtk_widget.GetRenderWindow().Render()
        self._update_chart()

    # ── IDW mesh coloring ─────────────────────────────────────────────────────

    def _recolor_mesh(self):
        """
        Color each brain-mesh vertex by IDW from electrode positions.

        Squared weights:  w_i = 1 / d_i²

        Vertices at distance < ε from an electrode receive that electrode's
        value directly (avoids division by zero and numerical instability).
        """
        if not self._electrode_positions or self._brain_poly is None:
            return

        mesh_pts = vtk_numpy.vtk_to_numpy(
            self._brain_poly.GetPoints().GetData()
        ).astype(np.float64)                                        # (N, 3)

        elec_pts = np.array(self._electrode_positions, dtype=np.float64)  # (E, 3)
        elec_val = np.array(self._electrode_values,    dtype=np.float64)  # (E,)

        # Pairwise distances  (N, E)
        diff  = mesh_pts[:, np.newaxis, :] - elec_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        EPS = 1e-6
        close_mask = dists < EPS   # (N, E) — vertex essentially ON an electrode

        # Squared inverse-distance weighting:  w_i = 1 / d_i²
        weights = np.where(close_mask, 1.0 / EPS**2, 1.0 / np.maximum(dists, EPS)**2)

        values = (weights * elec_val[np.newaxis, :]).sum(axis=1) / weights.sum(axis=1)

        # Override vertices sitting exactly on an electrode
        exact = close_mask.any(axis=1)
        if exact.any():
            electrode_idx = close_mask[exact].argmax(axis=1)
            values[exact] = elec_val[electrode_idx]

        # Write back via numpy (fast — avoids Python loop over all vertices)
        vtk_arr = vtk_numpy.numpy_to_vtk(
            values.astype(np.float32), deep=True, array_type=vtk.VTK_FLOAT
        )
        vtk_arr.SetName("EEG")
        self._brain_poly.GetPointData().SetScalars(vtk_arr)
        self._brain_poly.Modified()

    # ── Timer ─────────────────────────────────────────────────────────────────

    def _toggle_timer(self, on: bool):
        if on:
            self._qtimer.start()
            self.lbl_timer.setText("Timer: running")
        else:
            self._qtimer.stop()
            self.lbl_timer.setText("Timer: stopped")

    def _on_timer_tick(self):
        if not self._electrode_positions:
            return
        T = self.spin_T.value()
        for i in range(len(self._electrode_values)):
            val = random.choice(ELECTRODE_VALUES)
            self._electrode_values[i] = val
            self._signal_history[i].append(val)
            if len(self._signal_history[i]) > T:
                self._signal_history[i] = self._signal_history[i][-T:]

        self._recolor_mesh()
        self._update_chart()
        if hasattr(self, "_renderer"):
            self.vtk_widget.GetRenderWindow().Render()

    # ── Per-electrode signal chart ────────────────────────────────────────────

    def _update_chart(self):
        """
        Draw one subplot per electrode (2 rows × 4 cols).
        Each subplot shows the last T samples of that electrode's signal.
        """
        n = len(self._electrode_positions)
        T = self.spin_T.value()

        self.figure.clear()

        if n == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No electrodes placed yet",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.axis("off")
            self.figure.tight_layout()
            self.canvas.draw()
            return

        rows, cols = 2, 4
        for i in range(N_ELECTRODES):
            ax = self.figure.add_subplot(rows, cols, i + 1)
            ax.set_facecolor("#1a1a2e" if i >= n else "#0f0f1a")
            ax.set_ylim(-5, 105)
            ax.set_xlim(0, T - 1)
            ax.tick_params(labelsize=5)
            ax.set_title(f"E{i+1}", fontsize=7, color="white", pad=2)
            self.figure.patch.set_facecolor("#0f0f1a")

            if i < n and self._signal_history[i]:
                hist = self._signal_history[i][-T:]
                x    = list(range(len(hist)))
                ax.plot(x, hist, color=CHART_COLORS[i], linewidth=1.2,
                        marker="o", markersize=2)
                ax.axhline(hist[-1], color=CHART_COLORS[i], linewidth=0.5,
                           linestyle="--", alpha=0.5)
                ax.set_yticks([0, 50, 100])
                ax.tick_params(colors="white", labelsize=5)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444")
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "–", ha="center", va="center",
                        transform=ax.transAxes, color="#555", fontsize=9)

        self.figure.tight_layout(pad=0.4)
        self.canvas.draw()

    # ── Qt lifecycle ──────────────────────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        if not hasattr(self, "_renderer"):
            self._init_vtk()
        if self._reader is None and os.path.isfile(PATHS["head"]):
            self._setup_pipeline(PATHS["head"])
