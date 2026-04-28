"""
Task 2.1 – Basic Vascular Visualization
========================================
• Three orthogonal image planes (sagittal / transversal / coronal) with slice
  scrolling, controlled via Qt buttons and +/- keyboard shortcuts.
• Marching-cubes mesh for four tissue classes: skin, brain, grey matter, lesion.
• Smooth filter applied after marching cubes.
• Interactive visibility toggle for every actor.

Required data file: head_with_lesion.vtk
The iso-values below are reasonable starting points; adjust them once you have
inspected the actual scalar range of your dataset.
"""

import os
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QPushButton, QSlider, QLabel, QFileDialog, QSizePolicy, QCheckBox,
)
from PyQt5.QtCore import Qt

from .utils import PATHS, make_renderer, add_axes, build_isosurface


# ── Iso-value presets (scalar value → tissue type) ───────────────────────────
# head_with_lesion.vtk stores COLOR_SCALARS as unsigned char (0–255).
# The values below assume a typical label map:  0=bg, 1=skin, 2=brain,
# 3=grey matter, 4=lesion.  Marching cubes iso at label midpoints (0.5, 1.5 …).
#
# To inspect the actual labels in your dataset run:
#   python3 -c "
#   import vtk, numpy as np
#   r = vtk.vtkStructuredPointsReader(); r.SetFileName('…/head_with_lesion.vtk'); r.Update()
#   raw = vtk.util.numpy_support.vtk_to_numpy(r.GetOutput().GetPointData().GetScalars())
#   print(np.unique(raw))
#   "
# … then update the iso values here accordingly.
# Actual labels in head_with_lesion.vtk (uint8 COLOR_SCALARS):
#   0 = background, 42 = skin, 84 = skull, 127 = grey matter,
#   169 = brain (white matter), 254 = lesion
# Assignment requires four rendered structures (skin, grey matter, brain, lesion),
# so skull is intentionally not exposed as a separate mesh.
ISO_PRESETS = {
    #  name          iso     RGB colour            opacity
    # Skin: white + very transparent (inner structures visible through it)
    "skin":        ( 21,  (1.00, 1.00, 1.00), 0.20),   # boundary  0 ↔ 42
    "grey_matter": (105,  (0.70, 0.70, 0.95), 0.70),   # boundary 84 ↔ 127
    "brain":       (148,  (0.85, 0.72, 0.60), 1.00),   # boundary 127 ↔ 169
    "lesion":      (211,  (1.00, 0.10, 0.10), 1.00),   # boundary 169 ↔ 254
}

# Keyboard keys that toggle each mesh
MESH_KEYS = {
    "1": "skin",
    "2": "grey_matter",
    "3": "brain",
    "4": "lesion",
}


class Task1Widget(QWidget):
    """Main widget for Task 1: a VTK render window + Qt control panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reader = None
        self._plane_widgets: dict[str, vtk.vtkImagePlaneWidget] = {}
        self._mesh_actors: dict[str, vtk.vtkActor] = {}
        self._axes_widget = None
        self._active_plane: str = "transversal"   # last toggled-on plane (for +/- keys)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)

        # VTK render widget (left, takes 3/4 of width)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        root.addWidget(self.vtk_widget, 3)

        # Control panel (right)
        panel = QVBoxLayout()
        root.addLayout(panel, 1)

        # Load button
        btn_load = QPushButton("Load head_with_lesion.vtk")
        btn_load.clicked.connect(self._load_file)
        panel.addWidget(btn_load)

        # ── Image planes (one toggle + slider per plane) ──
        planes_box = QGroupBox("Image Planes")
        planes_layout = QVBoxLayout(planes_box)

        # Each entry: (attribute_name, label_text)
        plane_defs = [
            ("sagittal",    "Sagittal   (YZ)"),
            ("transversal", "Transversal (XY)"),
            ("coronal",     "Coronal    (XZ)"),
        ]

        self._plane_sliders: dict[str, QSlider] = {}
        self._plane_labels:  dict[str, QLabel]  = {}

        for key, label_text in plane_defs:
            # Toggle button
            btn = QPushButton(label_text)
            btn.setCheckable(True)
            btn.setChecked(False)
            setattr(self, f"btn_{key}", btn)
            planes_layout.addWidget(btn)

            # Slice label + horizontal slider on one row
            row = QHBoxLayout()
            lbl = QLabel("–")
            lbl.setFixedWidth(44)
            slider = QSlider(Qt.Horizontal)
            slider.setEnabled(False)
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(lbl)
            row.addWidget(slider)
            planes_layout.addLayout(row)

            self._plane_sliders[key] = slider
            self._plane_labels[key]  = lbl

        panel.addWidget(planes_box)

        # ── Meshes ──
        mesh_box = QGroupBox("Segmented Meshes")
        mesh_layout = QVBoxLayout(mesh_box)

        self.mesh_buttons: dict[str, QPushButton] = {}
        mesh_labels = {
            "skin":        "Skin  [key 1]",
            "grey_matter": "Grey Matter  [key 2]",
            "brain":       "Brain  [key 3]",
            "lesion":      "Lesion  [key 4]",
        }
        for key, label in mesh_labels.items():
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.setEnabled(False)
            mesh_layout.addWidget(btn)
            self.mesh_buttons[key] = btn

        # Opacity mode selector
        self.chk_transparent = QCheckBox("Transparent meshes")
        self.chk_transparent.setChecked(True)  # default: current assignment/report values
        self.chk_transparent.stateChanged.connect(lambda _: self._update_mesh_opacity_mode())
        mesh_layout.addWidget(self.chk_transparent)
        
        # Smoothing mode selector
        mesh_layout.addWidget(QLabel("Smoothing method:"))
        self.chk_smooth_image = QCheckBox("Smooth image before meshing\n(Gaussian, σ=1.0)")
        self.chk_smooth_mesh  = QCheckBox("Smooth mesh after meshing\n(Laplacian, 30 iter)")
        self.chk_smooth_mesh.setChecked(True)   # on by default, matches original behaviour
        self.chk_smooth_image.stateChanged.connect(lambda _: self._rebuild_meshes())
        self.chk_smooth_mesh.stateChanged.connect(lambda _:  self._rebuild_meshes())
        mesh_layout.addWidget(self.chk_smooth_image)
        mesh_layout.addWidget(self.chk_smooth_mesh)

        panel.addWidget(mesh_box)

        # ── Camera reset ──
        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self._reset_camera)
        panel.addWidget(btn_reset)

        panel.addStretch()

        # Info label
        self.info_label = QLabel(
            "Keyboard shortcuts (VTK focus):\n"
            "  s / t / c  toggle planes\n"
            "  +  /  -    scroll active plane\n"
            "  1-4  toggle meshes\n"
            "  r  reset camera"
        )
        self.info_label.setWordWrap(True)
        panel.addWidget(self.info_label)

    # ── VTK setup ─────────────────────────────────────────────────────────────

    def _init_vtk(self):
        """Initialise renderer and interactor (called once after Qt is shown)."""
        self._renderer = make_renderer()
        rw = self.vtk_widget.GetRenderWindow()
        rw.AddRenderer(self._renderer)
        self._iren = rw.GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self._iren.SetInteractorStyle(style)

        # Override key-press to handle +/- scrolling
        self._iren.AddObserver("KeyPressEvent", self._on_key_press)

        self._axes_widget = add_axes(self._renderer, self._iren)
        self.vtk_widget.Initialize()

    # ── File loading ──────────────────────────────────────────────────────────

    def _load_file(self):
        default = PATHS["head"] if os.path.isfile(PATHS["head"]) else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open head_with_lesion.vtk", default, "VTK files (*.vtk)"
        )
        if not path:
            return
        self._setup_pipeline(path)

    def _setup_pipeline(self, path: str):
        """Build the full VTK pipeline from the given file."""
        if not hasattr(self, "_renderer"):
            self._init_vtk()
        else:
            # Clean up previous actors/widgets
            self._renderer.RemoveAllViewProps()
            for pw in self._plane_widgets.values():
                pw.Off()
            self._plane_widgets.clear()
            self._mesh_actors.clear()

        # Reader
        self._reader = vtk.vtkStructuredPointsReader()
        self._reader.SetFileName(path)
        self._reader.Update()

        image = self._reader.GetOutput()
        dims  = image.GetDimensions()
        srange = image.GetScalarRange()
        print(f"[Task1] Loaded: dims={dims}  scalar range={srange}")

        # Gaussian image-smoothing filter (used when checkbox is on)
        self._gauss = vtk.vtkImageGaussianSmooth()
        self._gauss.SetInputConnection(self._reader.GetOutputPort())
        self._gauss.SetStandardDeviation(1.0)
        self._gauss.SetRadiusFactor(1.5)

        # Setup sub-components
        self._setup_image_planes(dims)
        self._setup_meshes()
        self._renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    # ── Image planes ──────────────────────────────────────────────────────────

    def _make_plane_widget(
        self, orientation: int, slice_idx: int
    ) -> vtk.vtkImagePlaneWidget:
        pw = vtk.vtkImagePlaneWidget()
        pw.SetInputConnection(self._reader.GetOutputPort())
        pw.SetPlaneOrientation(orientation)   # 0=YZ (sagittal), 1=XZ (coronal), 2=XY (transversal)
        pw.SetSliceIndex(slice_idx)
        pw.SetInteractor(self._iren)
        pw.TextureInterpolateOn()
        pw.SetResliceInterpolateToCubic()
        return pw

    def _setup_image_planes(self, dims):
        """Create three orthogonal plane widgets and wire each to its own slider."""
        W, H, D = dims

        # orientation: 0=YZ (sagittal, X-axis), 1=XZ (coronal, Y-axis), 2=XY (transversal, Z-axis)
        # n_slices is the number of slices along the plane's normal axis
        configs = {
            #  name          orient  mid_idx  n_slices
            "sagittal":    (0, W // 2, W),
            "transversal": (2, D // 2, D),
            "coronal":     (1, H // 2, H),
        }

        # Disconnect all previous signals to avoid stacking on reload
        for key in ("sagittal", "transversal", "coronal"):
            btn = getattr(self, f"btn_{key}")
            try:
                btn.toggled.disconnect()
            except TypeError:
                pass
            slider = self._plane_sliders[key]
            try:
                slider.valueChanged.disconnect()
            except TypeError:
                pass

        for name, (orient, mid, n_slices) in configs.items():
            pw = self._make_plane_widget(orient, mid)
            self._plane_widgets[name] = pw

            btn    = getattr(self, f"btn_{name}")
            slider = self._plane_sliders[name]
            lbl    = self._plane_labels[name]

            # Initialise slider range and value
            slider.setRange(0, n_slices - 1)
            slider.setValue(mid)
            slider.setEnabled(True)
            lbl.setText(str(mid))

            # Button: toggle plane visibility + track active plane for keyboard
            btn.toggled.connect(
                lambda on, n=name, p=pw: (
                    p.SetEnabled(int(on)),
                    setattr(self, "_active_plane", n) if on else None,
                    self.vtk_widget.GetRenderWindow().Render(),
                )
            )

            # Slider: move the corresponding plane widget
            slider.valueChanged.connect(
                lambda val, n=name, p=pw, lb=lbl: (
                    lb.setText(str(val)),
                    p.SetSliceIndex(val),
                    self.vtk_widget.GetRenderWindow().Render(),
                )
            )

    # ── Meshes ────────────────────────────────────────────────────────────────

    def _mesh_input_port(self):
        """Return the VTK output port to feed into the contour filter."""
        if self.chk_smooth_image.isChecked():
            return self._gauss.GetOutputPort()
        return self._reader.GetOutputPort()

    def _setup_meshes(self):
        """Build marching-cubes actors for all four tissue classes."""
        use_smooth_mesh = self.chk_smooth_mesh.isChecked()
        port = self._mesh_input_port()

        for name, (iso, color, opacity) in ISO_PRESETS.items():
            actor, _ = build_isosurface(
                port, iso, color, opacity,
                smooth_iterations=30 if use_smooth_mesh else 0,
            )
            actor.GetProperty().SetOpacity(self._target_opacity(name))
            actor.SetVisibility(False)
            self._renderer.AddActor(actor)
            self._mesh_actors[name] = actor

            btn = self.mesh_buttons[name]
            btn.setEnabled(True)
            btn.setChecked(False)
            try:
                btn.toggled.disconnect()
            except TypeError:
                pass
            btn.toggled.connect(
                lambda on, a=actor: (a.SetVisibility(int(on)),
                                     self.vtk_widget.GetRenderWindow().Render())
            )

    def _target_opacity(self, mesh_name: str) -> float:
        """Return opacity for the given mesh under the selected mode."""
        if self.chk_transparent.isChecked():
            return ISO_PRESETS[mesh_name][2]  # keep current per-mesh transparency
        return 1.0

    def _update_mesh_opacity_mode(self):
        """Apply current opacity mode to all existing mesh actors."""
        if not self._mesh_actors:
            return
        for name, actor in self._mesh_actors.items():
            actor.GetProperty().SetOpacity(self._target_opacity(name))
        self.vtk_widget.GetRenderWindow().Render()

    def _rebuild_meshes(self):
        """Remove existing mesh actors and rebuild with the current smoothing settings."""
        if self._reader is None:
            return
        for actor in self._mesh_actors.values():
            self._renderer.RemoveActor(actor)
        self._mesh_actors.clear()
        self._setup_meshes()
        self.vtk_widget.GetRenderWindow().Render()

    # ── Keyboard handler ──────────────────────────────────────────────────────

    def _on_key_press(self, obj, event):
        key = self._iren.GetKeySym()
        if key in ("plus", "equal"):
            self._scroll_active_plane(+1)
        elif key in ("minus", "underscore"):
            self._scroll_active_plane(-1)
        elif key == "s":
            self.btn_sagittal.setChecked(not self.btn_sagittal.isChecked())
        elif key == "t":
            self.btn_transversal.setChecked(not self.btn_transversal.isChecked())
        elif key == "c":
            self.btn_coronal.setChecked(not self.btn_coronal.isChecked())
        elif key in MESH_KEYS:
            name = MESH_KEYS[key]
            btn  = self.mesh_buttons.get(name)
            if btn and btn.isEnabled():
                btn.setChecked(not btn.isChecked())

    def _scroll_active_plane(self, delta: int):
        """Scroll the last-activated plane by *delta* slices via its slider."""
        name   = self._active_plane
        btn    = getattr(self, f"btn_{name}", None)
        slider = self._plane_sliders.get(name)
        if btn is None or slider is None or not btn.isChecked():
            # Fallback: find the first visible plane
            for n in ("transversal", "sagittal", "coronal"):
                b = getattr(self, f"btn_{n}")
                if b.isChecked():
                    name   = n
                    slider = self._plane_sliders[n]
                    break
            else:
                return
        new_val = max(slider.minimum(),
                      min(slider.maximum(), slider.value() + delta))
        slider.setValue(new_val)   # triggers valueChanged → plane + label update

    # ── Camera ────────────────────────────────────────────────────────────────

    def _reset_camera(self):
        if hasattr(self, "_renderer"):
            self._renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

    # ── Qt lifecycle ──────────────────────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        if not hasattr(self, "_renderer"):
            self._init_vtk()
        # Auto-load if default file exists
        if self._reader is None and os.path.isfile(PATHS["head"]):
            self._setup_pipeline(PATHS["head"])
