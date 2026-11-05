# SPDX-License-Identifier: Apache-2.0
"""Scripted 3D Slicer module that exports surface models to a USD file."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import qt
import slicer
import vtk
from MONAIPhysioLib.helper import MONAIPhysioHelper, MONAIPhysioLogic
from MONAIPhysioLib.runtime import RELEASE_URL
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

LAST_OUTPUT_KEY = "MONAIPhysio/lastOutput"
COLORS_KEY = "MONAIPhysio/colors"
MESH_KEY = "MONAIPhysio/meshType"
RESOLUTION_KEY = "MONAIPhysio/resolution"
# Scene of segmented chest and abdomen models (TotalSegmentator structures).
SAMPLE_MODELS = {
    "sampleName": "MONAI-Physio-Sample-Models",
    "uris": f"{RELEASE_URL}/MONAI-Physio-Sample-Models.mrb",
    "fileNames": "MONAI-Physio-Sample-Models.mrb",
    "nodeNames": "MONAI-Physio-Sample-Models",
    "checksums": "SHA256:6c070daf42bebe70c4dd3d7561e3125595e25fdb6f82ddf14c86235188ccce2f",
    "loadFileType": "SceneFile",
}


def register_sample_data() -> None:
    """List the sample models in Slicer's Sample Data module under Simulation."""
    import SampleData

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="Simulation",
        thumbnailFileName=str(
            Path(__file__).parent / "Resources/Icons/MONAIPhysioExportUSD.png"
        ),
        **SAMPLE_MODELS,
    )


class MONAIPhysioExportUSD(ScriptedLoadableModule):
    """Register the module in Slicer's Simulation category."""

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        parent.title = "MONAI Physio USD Export"
        parent.categories = ["Simulation"]
        parent.contributors = ["Stephen Aylward"]
        parent.dependencies = []
        parent.icon = qt.QIcon(str(
            Path(__file__).parent / "Resources/Icons/MONAIPhysioExportUSD.png"
        ))
        parent.helpText = (
            "Export the visible surface models (or one model) to a USD file "
            "with MONAI-Physio, using Slicer's Python, as surfaces or as "
            "tetrahedra or hexahedra of a chosen size. Apply writes the file; "
            "AutoRun re-writes it when parameters or models change. Open the "
            "exported asset in your Omniverse application.\n\n"
            "Requirements: surface and hexahedra export run on the CPU; "
            "tetrahedra need a CUDA GPU supported by the installed PyTorch.\n\n"
            "Downloads: 'Set up MONAI-Physio in Slicer...' asks first, then "
            "installs PyTorch (download.pytorch.org), torch-scatter "
            "(data.pyg.org or this extension's GitHub 'Data' release) and "
            "MONAI-Physio with its dependencies (PyPI) into Slicer's Python: "
            "several GB, so the first setup takes a while. 'Download sample "
            "models' fetches an 80 MB scene. Nothing else is downloaded or sent."
        )
        parent.acknowledgementText = "Built on MONAI-Physio, 3D Slicer and OpenUSD."
        slicer.app.connect("startupCompleted()", register_sample_data)


class MONAIPhysioExportUSDWidget(ScriptedLoadableModuleWidget):
    """Capture on Slicer's thread, then convert in an asynchronous worker.

    Laid out like Slicer's CLI modules: Apply writes the output file, and the
    AutoRun menu button re-runs on parameter or input changes.
    """

    UPDATE_DELAY_MS = 750

    def setup(self) -> None:
        super().setup()
        self.helper = MONAIPhysioHelper()
        self.pending = False
        self.observations: list[tuple[Any, int]] = []
        self.timer = qt.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._auto_run)
        self.helper.on_finished = self._run_finished
        self.helper.build_header(self.layout)
        download = qt.QPushButton("Download sample models")
        download.toolTip = (
            "Download and load a scene of segmented chest and abdomen models "
            "(80 MB, cached after the first download)"
        )
        download.clicked.connect(self._download_sample)
        # Right after the checkbox, before the row's trailing stretch.
        self.helper.export_all_row.insertWidget(1, download)
        self.helper.export_all.toggled.connect(self._parameter_changed)
        self.helper.model.currentNodeChanged.connect(self._parameter_changed)

        form = self.helper.section(self.layout, "USD output")
        self.colors = qt.QComboBox()
        self.colors.addItem("Slicer display colors", "slicer")
        self.colors.addItem("MONAI-Physio anatomy materials", "anatomy")
        self.colors.toolTip = (
            "Slicer: each model's current color, opacity and shading. "
            "Anatomy: MONAI-Physio's color and OmniSurface material, chosen "
            "from the model name (e.g. 'Left Lung' -> lung)."
        )
        mode = qt.QSettings().value(COLORS_KEY, "slicer")
        self.colors.currentIndex = max(0, self.colors.findData(mode))
        self.helper.colors = self._color_mode()
        self.colors.currentIndexChanged.connect(self._colors_changed)
        form.addRow("Colors and materials", self.colors)
        self.mesh_type = qt.QComboBox()
        self.mesh_type.addItem("Surface", "surface")
        self.mesh_type.addItem("Tetrahedra", "tetrahedra")
        self.mesh_type.addItem("Hexahedra", "hexahedra")
        self.mesh_type.toolTip = (
            "Surface: the model's own triangles. Tetrahedra or hexahedra: "
            "fill each closed model with voxel elements of the chosen "
            "resolution (a USD TetMesh, or a Mesh of the boundary quads with "
            "the cells in physio:hexVertexIndices). Tetrahedra are relaxed "
            "onto the smooth surface; hexahedra keep the voxel staircase."
        )
        mesh_type = qt.QSettings().value(MESH_KEY, "surface")
        self.mesh_type.currentIndex = max(0, self.mesh_type.findData(mesh_type))
        form.addRow("Mesh", self.mesh_type)
        self.resolution = qt.QDoubleSpinBox()
        self.resolution.decimals = 2
        self.resolution.minimum = 0.1
        self.resolution.maximum = 100.0
        self.resolution.singleStep = 0.5
        self.resolution.suffix = " mm"
        self.resolution.value = float(qt.QSettings().value(RESOLUTION_KEY, 2.0))
        self.resolution.toolTip = (
            "Edge length of the voxel elements; smaller is finer and slower"
        )
        form.addRow("Sampling resolution", self.resolution)
        self._mesh_changed()
        self.mesh_type.currentIndexChanged.connect(self._mesh_changed)
        self.resolution.valueChanged.connect(self._mesh_changed)
        self.output = qt.QLineEdit()
        self.output.placeholderText = "Choose a .usd, .usda or .usdc file"
        self.output.text = qt.QSettings().value(LAST_OUTPUT_KEY, "")
        self.output.editingFinished.connect(self._parameter_changed)
        browse = qt.QToolButton()
        browse.text = "..."
        browse.toolTip = "Choose the output USD file"
        browse.clicked.connect(self._choose_file)
        row = qt.QHBoxLayout()
        row.addWidget(self.output)
        row.addWidget(browse)
        form.addRow("Output USD file", row)

        self.helper.build_footer(self.layout, auto_run=True)
        self.helper.apply_button.toolTip = "Write the models to the output USD file"
        self.helper.apply_button.clicked.connect(self._apply)
        self.helper.auto_run_button.toggled.connect(self._auto_run_toggled)
        self.helper.auto_run_actions["on_input"].toggled.connect(
            lambda on: self._rebind()
        )

    def _download_sample(self) -> None:
        """Download (or reuse the cached) sample model scene and load it."""
        import SampleData

        try:
            with slicer.util.WaitCursor():
                SampleData.SampleDataLogic().downloadFromURL(
                    uris=SAMPLE_MODELS["uris"],
                    fileNames=SAMPLE_MODELS["fileNames"],
                    nodeNames=SAMPLE_MODELS["nodeNames"],
                    checksums=SAMPLE_MODELS["checksums"],
                    loadFileTypes=SAMPLE_MODELS["loadFileType"],
                    # SampleData returns scene files unloaded unless asked.
                    loadFiles=True,
                )
        except Exception as exc:
            slicer.util.errorDisplay(f"Could not download the sample models: {exc}")
            return
        count = len(self.helper.logic.visible_models())
        self.helper.log.appendPlainText(
            f"Loaded the sample models: {count} visible surface models to export."
        )

    def _color_mode(self) -> str:
        return self.colors.itemData(self.colors.currentIndex) or "slicer"

    def _colors_changed(self, index: Any = None) -> None:
        self.helper.colors = self._color_mode()
        qt.QSettings().setValue(COLORS_KEY, self.helper.colors)
        self._parameter_changed()

    def _mesh_changed(self, *args: Any) -> None:
        mesh_type = self.mesh_type.itemData(self.mesh_type.currentIndex) or "surface"
        self.resolution.enabled = mesh_type != "surface"
        self.helper.mesh_type = mesh_type
        self.helper.resolution = self.resolution.value
        settings = qt.QSettings()
        settings.setValue(MESH_KEY, mesh_type)
        settings.setValue(RESOLUTION_KEY, self.resolution.value)
        if args:
            self._parameter_changed()

    def _target(self) -> Optional[Path]:
        text = self.output.text.strip()
        return Path(text) if text else None

    def _choose_file(self) -> bool:
        """Ask where to save, starting at the current or last saved file."""
        start = self.output.text.strip() or qt.QSettings().value(
            LAST_OUTPUT_KEY, ""
        )
        path = qt.QFileDialog.getSaveFileName(
            self.parent, "Save USD", start or "model.usdc",
            "USD (*.usd *.usda *.usdc)",
        )
        if not path:
            return False
        self.output.text = path
        qt.QSettings().setValue(LAST_OUTPUT_KEY, str(path))
        self._parameter_changed()
        return True

    def _apply(self) -> None:
        """Write the output file, asking for one first if none is set."""
        if self._target() is None and not self._choose_file():
            return
        qt.QSettings().setValue(LAST_OUTPUT_KEY, str(self._target()))
        self.helper.run(self._target())

    # AutoRun: re-run on parameter changes and/or on model, display, transform
    # and scene changes; debounce a burst of events into one export, and either
    # wait for or cancel a running export.
    def _auto_run_toggled(self, enabled: bool) -> None:
        self.timer.stop()
        self.pending = False
        if enabled and self._target() is None and not self._choose_file():
            self.helper.auto_run_button.checked = False
            return
        self._rebind()
        if enabled:
            self._schedule()

    def _rebind(self) -> None:
        self._unobserve()
        if self.helper.auto_run_button.checked and self.helper.auto_run_option(
            "on_input"
        ):
            self._observe()

    def _observe(self) -> None:
        scene = slicer.mrmlScene
        for event in (scene.NodeAddedEvent, scene.NodeRemovedEvent):
            self.observations.append(
                (scene, scene.AddObserver(event, self._scene_changed))
            )
        self._observe_models()

    def _observe_models(self) -> None:
        for model in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            for event in (
                vtk.vtkCommand.ModifiedEvent,
                model.PolyDataModifiedEvent,
                model.TransformModifiedEvent,
            ):
                self.observations.append(
                    (model, model.AddObserver(event, self._input_changed))
                )
            display = model.GetDisplayNode()
            if display is not None:
                self.observations.append((display, display.AddObserver(
                    vtk.vtkCommand.ModifiedEvent, self._input_changed
                )))

    def _unobserve(self) -> None:
        for node, tag in self.observations:
            node.RemoveObserver(tag)
        self.observations = []

    def _scene_changed(self, caller: Any = None, event: Any = None) -> None:
        # Nodes came or went: rebind model observers, then export the new state.
        scene_tags = [o for o in self.observations if o[0] is slicer.mrmlScene]
        for node, tag in self.observations:
            if node is not slicer.mrmlScene:
                node.RemoveObserver(tag)
        self.observations = scene_tags
        self._observe_models()
        self._schedule()

    def _input_changed(self, caller: Any = None, event: Any = None) -> None:
        self._schedule()

    def _parameter_changed(self, *args: Any) -> None:
        if self.helper.auto_run_option("on_parameter"):
            self._schedule()

    def _schedule(self) -> None:
        button = getattr(self.helper, "auto_run_button", None)
        if button is not None and button.checked and self._target() is not None:
            self.timer.start(self.UPDATE_DELAY_MS)

    def _auto_run(self) -> None:
        if self.helper.process is not None:
            self.pending = True
            if self.helper.auto_run_option("cancels"):
                self.helper.cancel()
        elif self.helper.auto_run_button.checked and self._target() is not None:
            self.helper.run(self._target(), quiet=True)

    def _run_finished(self, success: bool) -> None:
        if self.pending:
            self.pending = False
            self._schedule()

    def cleanup(self) -> None:
        self.timer.stop()
        self._unobserve()
        self.helper.cleanup()


class MONAIPhysioExportUSDTest(ScriptedLoadableModuleTest):
    """Run a real MRML capture smoke test inside Slicer without external SDKs."""

    def runTest(self) -> None:
        self.test_model_capture()

    def test_model_capture(self) -> None:
        """Verify a transformed sphere and source immutability in Slicer."""
        import json

        import vtk

        source = vtk.vtkSphereSource()
        source.Update()
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        try:
            model.SetAndObservePolyData(source.GetOutput())
            model.CreateDefaultDisplayNodes()
            model.GetDisplayNode().SetColor(0.7, 0.2, 0.1)
            model.GetDisplayNode().SetOpacity(0.6)
            matrix = vtk.vtkMatrix4x4()
            matrix.Identity()
            matrix.SetElement(0, 3, 100.0)
            transform.SetMatrixTransformToParent(matrix)
            model.SetAndObserveTransformNodeID(transform.GetID())
            before = model.GetPolyData().GetPoint(0)
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                logic = MONAIPhysioLogic()
                self.assertIn(model, logic.visible_models())
                snapshot = logic.capture(model, work)
                metadata = json.loads(snapshot.read_text(encoding="utf-8"))
                self.assertAlmostEqual(metadata["display"]["opacity"], 0.6)
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(str(work / "model.vtp"))
                reader.Update()
                self.assertAlmostEqual(
                    reader.GetOutput().GetPoint(0)[0], before[0] + 100.0, places=4
                )
                self.assertEqual(model.GetPolyData().GetPoint(0), before)
                arguments = logic.worker_arguments([snapshot], work / "model.usdc")
                self.assertTrue(Path(arguments[1]).is_file())
        finally:
            display = model.GetDisplayNode()
            slicer.mrmlScene.RemoveNode(model)
            if display:
                slicer.mrmlScene.RemoveNode(display)
            slicer.mrmlScene.RemoveNode(transform)
