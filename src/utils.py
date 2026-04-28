"""
Shared utilities: data paths, VTK pipeline helpers, and Qt/VTK widget factory.
"""

import os
import vtk
import numpy as np
from vtkmodules.util import numpy_support as vtk_numpy

# ── Data paths ────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, "Neuroimaging_visualization_project")

PATHS = {
    "head":      os.path.join(DATA_DIR, "head_with_lesion.vtk"),
    "vessels":   os.path.join(DATA_DIR, "vessels_data.vtk"),
    "dsa_dir":   os.path.join(DATA_DIR, "DSA of AVM examples"),
    "dsa1":      os.path.join(DATA_DIR, "DSA of AVM examples", "vdm01_1.vtk"),
    "dsa2":      os.path.join(DATA_DIR, "DSA of AVM examples", "vdm02_1.vtk"),
}


def check_data_files() -> list[str]:
    """Return list of missing data files (head_with_lesion.vtk is optional for tasks 1/2)."""
    missing = []
    for key, path in PATHS.items():
        if key == "dsa_dir":
            continue
        if not os.path.isfile(path):
            missing.append(f"{key}: {path}")
    return missing


# ── VTK reader helpers ────────────────────────────────────────────────────────

def read_vtk_structured_points(path: str) -> vtk.vtkStructuredPointsReader:
    """Read a legacy VTK structured-points file and return an updated reader."""
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(path)
    reader.Update()
    return reader


def read_vtk_structured_points_as_image(path: str) -> vtk.vtkImageData:
    """Return vtkImageData from a legacy VTK structured-points file."""
    reader = read_vtk_structured_points(path)
    # vtkStructuredPointsReader output can be cast to vtkImageData
    cast = vtk.vtkImageCast()
    cast.SetInputConnection(reader.GetOutputPort())
    cast.SetOutputScalarTypeToFloat()
    cast.Update()
    return cast.GetOutput()


# ── Marching-cubes mesh builder ───────────────────────────────────────────────

def build_isosurface(
    reader_port,
    iso_value: float,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    smooth_iterations: int = 30,
    smooth_relaxation: float = 0.1,
) -> tuple[vtk.vtkActor, vtk.vtkContourFilter]:
    """
    Run marching cubes + smoothing on *reader_port* and return (actor, contour).
    The caller is responsible for adding the actor to a renderer.
    """
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(reader_port)
    contour.SetValue(0, iso_value)

    if smooth_iterations > 0:
        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(contour.GetOutputPort())
        smooth.SetNumberOfIterations(smooth_iterations)
        smooth.SetRelaxationFactor(smooth_relaxation)
        normals_input = smooth.GetOutputPort()
    else:
        normals_input = contour.GetOutputPort()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(normals_input)
    normals.SetFeatureAngle(60.0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)

    return actor, contour


# ── Renderer / render-window helpers ─────────────────────────────────────────

def make_renderer(background=(0.1, 0.1, 0.2)) -> vtk.vtkRenderer:
    ren = vtk.vtkRenderer()
    ren.SetBackground(*background)
    return ren


def add_axes(renderer: vtk.vtkRenderer, interactor: vtk.vtkRenderWindowInteractor):
    """Overlay a small orientation-axes widget in the bottom-left corner."""
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.15, 0.15)
    widget.EnabledOn()
    widget.InteractiveOff()
    return widget


# ── Colormap / lookup-table helpers ──────────────────────────────────────────

def make_rainbow_lut(range_min: float = 0, range_max: float = 100) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.667, 0.0)   # blue → red
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetValueRange(1.0, 1.0)
    lut.SetRange(range_min, range_max)
    lut.SetNumberOfColors(256)
    lut.Build()
    return lut


def make_time_lut(n_frames: int) -> vtk.vtkLookupTable:
    """Map frame index [0, n_frames-1] to a rainbow color (first=red, last=blue)."""
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.0, 0.667)   # red → blue
    lut.SetRange(0, n_frames - 1)
    lut.SetNumberOfColors(256)
    lut.Build()
    return lut


# ── numpy ↔ VTK image conversion ─────────────────────────────────────────────

def numpy_to_vtk_image(
    array: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
) -> vtk.vtkImageData:
    """
    Convert a 2-D or 3-D numpy array (H, W[, C]) to vtkImageData.
    For RGB/RGBA arrays the last axis is treated as components.
    Array dtype is preserved.
    """
    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    h, w, c = array.shape
    vtk_array = vtk_numpy.numpy_to_vtk(
        array.ravel(order="C"), deep=True,
        array_type=vtk_numpy.get_vtk_array_type(array.dtype),
    )
    vtk_array.SetNumberOfComponents(c)

    img = vtk.vtkImageData()
    img.SetDimensions(w, h, 1)
    img.SetSpacing(*spacing)
    img.SetOrigin(*origin)
    img.GetPointData().SetScalars(vtk_array)
    return img
