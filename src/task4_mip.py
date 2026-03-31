"""
Task 2.4 – Slicing with Maximum Intensity Projection
=====================================================
Maximum Intensity Projection (MIP) visualises volumetric data by casting
parallel rays through the volume and retaining the maximum voxel intensity
encountered along each ray.  The result is a 2-D image that reveals
high-intensity structures.

Plain MIP collapses the entire volume into one image, losing depth
information.  Sliced MIP addresses this: instead of projecting the full
volume, only a slab of N slices (default: 10 % of the total depth in each
direction) is projected.

Three orthogonal views are displayed simultaneously:
    Transverse    – projection along Z, slab scrolled in Z
    Sagittal – projection along X, slab scrolled in X
    Coronal  – projection along Y, slab scrolled in Y

Window/Level controls map the projection intensities to the display grey scale.

Data: vessels_data.vtk
"""

import os
import vtk
import numpy as np
from vtkmodules.util import numpy_support as vtk_numpy
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QSlider, QLabel, QSizePolicy,
)
from PyQt5.QtCore import Qt

from .utils import PATHS, make_renderer, numpy_to_vtk_image


# Projection axis index for each named view
AXES: dict[str, int] = {
    "Transverse (Z)":    2,
    "Sagittal (X)":  0,
    "Coronal (Y)":  1,
}


class Task4Widget(QWidget):
    """
    Task 4: interactive sliced-MIP viewer for vessels_data.vtk.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._reader:    vtk.vtkStructuredPointsReader | None = None
        self._vol:       np.ndarray | None = None   # (Z, Y, X) float32 volume

        # Per-view VTK objects updated by the sliders
        self._wl_filters: dict[str, vtk.vtkImageMapToWindowLevelColors] = {}
        self._vtk_images: dict[str, vtk.vtkImageData]  = {}
        self._renderers:  dict[str, vtk.vtkRenderer]   = {}
        self._vtk_widgets: dict[str, QVTKRenderWindowInteractor] = {}

        self._build_ui()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ── Three VTK viewports arranged in a 2×2 grid ───────────────────────
        vtk_panel = QWidget()
        grid = QGridLayout(vtk_panel)
        grid.setSpacing(2)
        root.addWidget(vtk_panel, stretch=3)

        positions = {
            "Transverse (Z)":    (0, 0),
            "Sagittal (X)":  (0, 1),
            "Coronal (Y)":  (1, 0),
        }
        for name, (row, col) in positions.items():
            w = QVTKRenderWindowInteractor(vtk_panel)
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            grid.addWidget(w, row, col)
            self._vtk_widgets[name] = w

        # ── Control panel ─────────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(280)
        cl = QVBoxLayout(ctrl)
        root.addWidget(ctrl)

        btn_load = QPushButton("Load vessels_data.vtk")
        btn_load.clicked.connect(self._load_file)
        cl.addWidget(btn_load)

        # Slab thickness (percentage of total slices in each direction)
        thick_box = QGroupBox("Slab thickness")
        tl = QVBoxLayout(thick_box)
        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("1 %"))
        self._slider_thick = QSlider(Qt.Horizontal)
        self._slider_thick.setRange(1, 50)
        self._slider_thick.setValue(10)
        self._slider_thick.valueChanged.connect(self._on_thickness_changed)
        row_t.addWidget(self._slider_thick)
        row_t.addWidget(QLabel("50 %"))
        tl.addLayout(row_t)
        self._lbl_thick = QLabel("Thickness: 10 %")
        tl.addWidget(self._lbl_thick)
        cl.addWidget(thick_box)

        # Per-view position sliders
        self._sliders_pos: dict[str, QSlider] = {}
        self._labels_pos:  dict[str, QLabel]  = {}

        for name in AXES:
            box = QGroupBox(f"Position — {name}")
            bl  = QVBoxLayout(box)
            row_p = QHBoxLayout()
            lbl = QLabel("–")
            lbl.setFixedWidth(36)
            sl = QSlider(Qt.Horizontal)
            sl.setEnabled(False)
            row_p.addWidget(lbl)
            row_p.addWidget(sl)
            bl.addLayout(row_p)
            cl.addWidget(box)
            self._sliders_pos[name] = sl
            self._labels_pos[name]  = lbl

        # Window / Level
        wl_box = QGroupBox("Window / Level")
        wl_l = QVBoxLayout(wl_box)

        wl_l.addWidget(QLabel("Window (contrast):"))
        self._slider_window = QSlider(Qt.Horizontal)
        self._slider_window.setRange(1, 4000)
        self._slider_window.setValue(1000)
        self._slider_window.setEnabled(False)
        self._slider_window.valueChanged.connect(self._on_wl_changed)
        wl_l.addWidget(self._slider_window)

        wl_l.addWidget(QLabel("Level (brightness):"))
        self._slider_level = QSlider(Qt.Horizontal)
        self._slider_level.setRange(-1000, 3000)
        self._slider_level.setValue(500)
        self._slider_level.setEnabled(False)
        self._slider_level.valueChanged.connect(self._on_wl_changed)
        wl_l.addWidget(self._slider_level)

        self._lbl_wl = QLabel("W: 1000  L: 500")
        wl_l.addWidget(self._lbl_wl)
        cl.addWidget(wl_box)

        btn_reset = QPushButton("Reset cameras")
        btn_reset.clicked.connect(self._reset_cameras)
        cl.addWidget(btn_reset)

        cl.addStretch()
        cl.addWidget(QLabel(
            "Three simultaneous MIP views.\n"
            "Each view projects a slab of slices\n"
            "along its own axis.\n\n"
            "Use the position sliders to scroll\n"
            "the slab through the volume."
        ))

    # =========================================================================
    # VTK initialisation
    # =========================================================================

    def _init_vtk(self) -> None:
        for name, vtk_widget in self._vtk_widgets.items():
            ren = make_renderer(background=(0.05, 0.05, 0.05))
            rw  = vtk_widget.GetRenderWindow()
            rw.AddRenderer(ren)

            iren = rw.GetInteractor()
            iren.SetInteractorStyle(vtk.vtkInteractorStyleImage())

            self._renderers[name] = ren
            vtk_widget.Initialize()

    # =========================================================================
    # Data loading
    # =========================================================================

    def _load_file(self) -> None:
        from PyQt5.QtWidgets import QFileDialog
        default = PATHS["vessels"] if os.path.isfile(PATHS["vessels"]) else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open vessels_data.vtk", default, "VTK files (*.vtk)"
        )
        if path:
            self._setup_pipeline(path)

    def _setup_pipeline(self, path: str) -> None:
        """
        Read the volume, cache it as a NumPy array, and build the initial
        MIP image for each view.

        """
        if not self._renderers:
            self._init_vtk()

        self._reader = vtk.vtkStructuredPointsReader()
        self._reader.SetFileName(path)
        self._reader.Update()

        image = self._reader.GetOutput()
        dims  = image.GetDimensions()   # (Nx, Ny, Nz)
        s_min, s_max = image.GetScalarRange()

        # Convert to (Z, Y, X) NumPy array
        flat = vtk_numpy.vtk_to_numpy(
            image.GetPointData().GetScalars()
        ).astype(np.float32)
        Nx, Ny, Nz = dims
        self._vol = flat.reshape(Nz, Ny, Nx)   # shape: (Z, Y, X)

        # Initialise Window/Level from the actual scalar range
        data_range  = s_max - s_min
        init_window = max(1, int(data_range * 0.8))
        init_level  = int(s_min + data_range * 0.5)

        self._slider_window.setValue(init_window)
        self._slider_level.setValue(
            int(np.clip(init_level,
                        self._slider_level.minimum(),
                        self._slider_level.maximum()))
        )
        self._slider_window.setEnabled(True)
        self._slider_level.setEnabled(True)

        # Build initial MIP image and actor for every view
        for name, axis_idx in AXES.items():
            self._build_view(name, axis_idx, dims, init_window, init_level)

        # Configure position sliders after views are built
        self._configure_position_sliders(dims)

        self._render_all()

    # =========================================================================
    # MIP computation
    # =========================================================================

    def _compute_mip(self, name: str, start: int) -> np.ndarray:
        """
        Extract a slab from self._vol and return its MIP as a 2-D array.

        Returns a 2-D float32 array whose shape matches the display plane:
            Transverse    → (Y, X)
            Coronal  → (Z, X)
            Sagittal → (Z, Y)
        """
        Nz, Ny, Nx = self._vol.shape
        axis_idx   = AXES[name]
        n_slices   = [Nz, Ny, Nx][axis_idx]   # depth along projection axis
        slab_size  = max(1, int(n_slices * self._slider_thick.value() / 100))

        end = min(start + slab_size, n_slices)   # exclusive upper bound

        if axis_idx == 2:       # Transverse: project along Z
            slab = self._vol[start:end, :, :]    # (slab, Y, X)
            return slab.max(axis=0)              # (Y, X)
        elif axis_idx == 1:     # Coronal: project along Y
            slab = self._vol[:, start:end, :]    # (Z, slab, X)
            return slab.max(axis=1)              # (Z, X)
        else:                   # Sagittal: project along X
            slab = self._vol[:, :, start:end]    # (Z, Y, slab)
            return slab.max(axis=2)              # (Z, Y)

    # =========================================================================
    # View construction and update
    # =========================================================================

    def _build_view(
        self,
        name:      str,
        axis_idx:  int,
        dims:      tuple,
        window:    int,
        level:     int,
    ) -> None:

        Nx, Ny, Nz = dims
        n_slices   = [Nx, Ny, Nz][axis_idx]
        slab_size  = max(1, int(n_slices * self._slider_thick.value() / 100))
        mid_start  = (n_slices - slab_size) // 2

        mip = self._compute_mip(name, mid_start)   # 2-D (H, W) float32

        # Wrap the MIP array as vtkImageData using the shared utility.
        # numpy_to_vtk_image expects shape (H, W) or (H, W, C).
        vtk_img = numpy_to_vtk_image(mip)
        self._vtk_images[name] = vtk_img

        # Window/Level filter maps float intensities to display grey values
        wl = vtk.vtkImageMapToWindowLevelColors()
        wl.SetInputData(vtk_img)
        wl.SetWindow(window)
        wl.SetLevel(level)
        self._wl_filters[name] = wl

        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputConnection(wl.GetOutputPort())

        ren = self._renderers[name]
        ren.RemoveAllViewProps()
        ren.AddActor(actor)

        # View label in the lower-left corner
        txt = vtk.vtkTextActor()
        txt.SetInput(name)
        txt.GetTextProperty().SetFontSize(14)
        txt.GetTextProperty().SetColor(0.85, 0.85, 0.4)
        txt.SetPosition(8, 8)
        ren.AddActor2D(txt)

        ren.ResetCamera()

    def _update_mip(self, name: str, start: int) -> None:
        """
        Recompute the MIP for *name* at slab position *start* and push the
        result into the existing vtkImageData.
        """
        mip = self._compute_mip(name, start)   # (H, W) float32

        vtk_img = self._vtk_images[name]

        # Overwrite the scalar array in-place
        new_scalars = vtk_numpy.numpy_to_vtk(
            mip.ravel(), deep=True, array_type=vtk.VTK_FLOAT
        )
        vtk_img.GetPointData().SetScalars(new_scalars)
        vtk_img.Modified()

        self._wl_filters[name].Update()
        self._renderers[name].GetRenderWindow().Render()

    # =========================================================================
    # Slider configuration
    # =========================================================================

    def _configure_position_sliders(self, dims: tuple) -> None:
        """
        Set the range and initial value of each position slider.
        """
        Nx, Ny, Nz = dims
        n_slices_for = {
            "Transverse (Z)":    Nz,
            "Sagittal (X)":  Nx,
            "Coronal (Y)":  Ny,
        }
        for name, n_slices in n_slices_for.items():
            slab_size = max(1, int(n_slices * self._slider_thick.value() / 100))
            max_start = max(0, n_slices - slab_size)
            mid_start = max_start // 2

            sl = self._sliders_pos[name]
            try:
                sl.valueChanged.disconnect()
            except TypeError:
                pass

            sl.setRange(0, max_start)
            sl.setValue(mid_start)
            sl.setEnabled(True)
            self._labels_pos[name].setText(str(mid_start))

            sl.valueChanged.connect(
                lambda val, n=name: self._on_position_changed(n, val)
            )

    # =========================================================================
    # Slider callbacks
    # =========================================================================

    def _on_position_changed(self, name: str, val: int) -> None:
        """Scroll the slab of view *name* to position *val*."""
        self._labels_pos[name].setText(str(val))
        self._update_mip(name, val)

    def _on_thickness_changed(self, val: int) -> None:
        """
        Change the slab thickness for all views simultaneously.
        """
        self._lbl_thick.setText(f"Thickness: {val} %")
        if self._vol is None:
            return

        Nz, Ny, Nx = self._vol.shape
        n_slices_for = {
            "Transverse (Z)":    Nz,
            "Sagittal (X)":  Nx,
            "Coronal (Y)":  Ny,
        }
        for name, n_slices in n_slices_for.items():
            slab_size  = max(1, int(n_slices * val / 100))
            max_start  = max(0, n_slices - slab_size)

            sl = self._sliders_pos[name]
            old_centre = sl.value() + slab_size // 2
            new_start  = int(np.clip(old_centre - slab_size // 2, 0, max_start))

            try:
                sl.valueChanged.disconnect()
            except TypeError:
                pass
            sl.setRange(0, max_start)
            sl.setValue(new_start)
            sl.valueChanged.connect(
                lambda v, n=name: self._on_position_changed(n, v)
            )

            self._update_mip(name, new_start)

    def _on_wl_changed(self, _=None) -> None:
        """Apply the current Window/Level values to all three views."""
        w = self._slider_window.value()
        l = self._slider_level.value()
        self._lbl_wl.setText(f"W: {w}  L: {l}")
        for wl in self._wl_filters.values():
            wl.SetWindow(w)
            wl.SetLevel(l)
            wl.Modified()
        self._render_all()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _render_all(self) -> None:
        for vtk_w in self._vtk_widgets.values():
            vtk_w.GetRenderWindow().Render()

    def _reset_cameras(self) -> None:
        for ren in self._renderers.values():
            ren.ResetCamera()
        self._render_all()

    # =========================================================================
    # Qt lifecycle
    # =========================================================================

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._renderers:
            self._init_vtk()
        if self._reader is None and os.path.isfile(PATHS["vessels"]):
            self._setup_pipeline(PATHS["vessels"])
