"""
Neuroimaging Visualization – Main Entry Point
==============================================
A PyQt5 tabbed application that hosts all four project tasks:

  Tab 1 – Basic Vascular Visualization  (Task 2.1)
  Tab 2 – EEG Visualization             (Task 2.2)
  Tab 3 – DSA Visualization             (Task 2.3)
  Tab 4 – MIP Slicing                   (Task 2.4)

Run with:
    python main.py
"""

import sys
import vtk

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QStatusBar, QLabel,
)
from PyQt5.QtCore import Qt

from src.task1_basic_viz import Task1Widget
from src.task2_eeg_viz   import Task2Widget
from src.task3_dsa_viz_threshold   import Task3Widget
from src.task4_mip       import Task4Widget
from src.utils           import check_data_files


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuroimaging Visualization  –  E016712A Computer Graphics")
        self.resize(1400, 900)

        # ── Tab widget ────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Instantiate task widgets (VTK is lazy-initialised on first show)
        self._task1 = Task1Widget()
        self._task2 = Task2Widget()
        self._task3 = Task3Widget()
        self._task4 = Task4Widget()

        self.tabs.addTab(self._task1, "1 – Basic Visualization")
        self.tabs.addTab(self._task2, "2 – EEG Visualization")
        self.tabs.addTab(self._task3, "3 – DSA Visualization")
        self.tabs.addTab(self._task4, "4 – MIP Slicing")

        # ── Status bar ────────────────────────────────────────────────────────
        status = QStatusBar()
        self.setStatusBar(status)

        missing = check_data_files()
        if missing:
            msg = "Missing data: " + "  |  ".join(m.split(":")[0] for m in missing)
            lbl = QLabel(msg)
            lbl.setStyleSheet("color: orange;")
            status.addPermanentWidget(lbl)
        else:
            status.showMessage("All data files found.", 5000)

        vtk_ver = vtk.vtkVersion.GetVTKVersion()
        status.addPermanentWidget(QLabel(f"VTK {vtk_ver}"))


def main():
    # Route VTK messages to stderr instead of a separate popup window
    vtk.vtkOutputWindow.GetInstance().SetDisplayMode(
        vtk.vtkOutputWindow.NEVER
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
