# SPDX-License-Identifier: Apache-2.0
"""Scripted 3D Slicer module simulating respiratory lung motion or cardiac
heart motion from one CT."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)

# The shared MONAIPhysioLib ships with MONAIPhysioExportUSD. Installed modules
# share one directory; from a source checkout add the sibling module directory.
_SIBLING = Path(__file__).resolve().parent.parent / "MONAIPhysioExportUSD"
if _SIBLING.is_dir() and str(_SIBLING) not in sys.path:
    sys.path.append(str(_SIBLING))

from MONAIPhysioLib.helper import MONAIPhysioHelper  # noqa: E402
from MONAIPhysioLib.runtime import RELEASE_URL, python_slicer  # noqa: E402

# Ungated chest CT, the same volume MONAI-Physio's tutorial 0 downloads.
CHEST_CT = {
    "sampleName": "Chest-CT",
    "uris": f"{RELEASE_URL}/Chest-CT.mha",
    "fileNames": "Chest-CT.mha",
    "nodeNames": "Chest-CT",
    "checksums": "SHA256:e05ba4271997130531ed6e31c817c441e5f06f41c51e6124e60df1a66c99c419",
    "loadFileType": "VolumeFile",
}


def register_sample_data() -> None:
    """List the chest CT in Slicer's Sample Data module under Simulation."""
    import SampleData

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="Simulation",
        thumbnailFileName=str(
            Path(__file__).parent
            / "Resources/Icons/MONAIPhysioRespiratorySimulation.png"
        ),
        **CHEST_CT,
    )


SETTINGS ="MONAIPhysio/respiratorySimulation/"
DEFAULT_FOLDER = Path.home() / "MONAIPhysioRespiratorySimulation"
# Slicer's GenericAnatomyColors lung and heart colors, if the color table is
# unavailable.
LUNG_COLOR = (197 / 255, 165 / 255, 145 / 255)
HEART_COLOR = (206 / 255, 110 / 255, 84 / 255)

# What differs between the two simulations. "rate_setting" keeps the
# respiratory rate under its original settings key.
SIMULATIONS: dict[str, dict[str, Any]] = {
    "respiratory": {
        "label": "Respiratory (lungs)",
        "worker": "respiratory_simulation_worker.py",
        "weights": "physicsnemo_mgn_lung_motion",
        "network": "lung-motion",
        "color_name": "left_lung",
        "color": LUNG_COLOR,
        "default_rate": 12.0,
        "rate_setting": "cyclesPerMinute",
        "rate_label": "Breaths per minute",
        "surfaces": "Lung surfaces",
        "images": "Lung 4D-CT",
        "browser": "Respiratory motion",
        "stage": "respiratory stage",
        "usd_suffix": "respiration",
        "organ": "lung",
    },
    "cardiac": {
        "label": "Cardiac (heart)",
        "worker": "cardiac_simulation_worker.py",
        "weights": "physicsnemo_mgn_duke_heart_motion",
        "network": "heart-motion",
        "color_name": "heart",
        "color": HEART_COLOR,
        "default_rate": 60.0,
        "rate_setting": "cardiacCyclesPerMinute",
        "rate_label": "Beats per minute",
        "surfaces": "Heart surfaces",
        "images": "Heart 4D-CT",
        "browser": "Cardiac motion",
        "stage": "cardiac stage",
        "usd_suffix": "heartbeat",
        "organ": "heart",
    },
}
HEART_SEGMENTERS = (
    ("simpleware", "Simpleware Medical ASCardio (trained on)"),
    ("nvsegment", "NV-Segment-CTMR (approximate)"),
)


class MONAIPhysioRespiratorySimulation(ScriptedLoadableModule):
    """Register the module in Slicer's Simulation category."""

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        parent.title = "MONAI-Physio Respiratory and Cardiac Simulation"
        parent.categories = ["Simulation"]
        parent.contributors = ["Stephen Aylward"]
        parent.dependencies = []
        parent.icon = qt.QIcon(str(
            Path(__file__).parent
            / "Resources/Icons/MONAIPhysioRespiratorySimulation.png"
        ))
        parent.helpText = (
            "Simulates respiratory lung motion or cardiac heart motion for a "
            "chest CT loaded in Slicer, following MONAI-Physio's tutorial 0 "
            "notebooks. Respiratory: segments the lung lobes with "
            "NV-Segment-CTMR (needs a CUDA GPU), fits a lung shape model and "
            "predicts a breathing cycle with a pretrained network. Cardiac: "
            "segments the heart with Simpleware Medical ASCardio (a separate "
            "commercial install, the segmenter the heart model was built from) "
            "or, approximately, with NV-Segment-CTMR, fits the Duke heart shape "
            "model and predicts a heartbeat with a pretrained network. Writes "
            "the surfaces (VTP/) and optionally a warped 4D-CT (CT/) and an "
            "animated USD to one output folder; results load as Slicer "
            "sequences.\n\n"
            "Requirements: a CUDA GPU supported by the installed PyTorch, "
            "with 16 GB or more of memory (about 15 GB used in testing), and "
            "about 40 GB of free system memory (RAM plus page file) for a "
            "512x512 chest CT: the lung shape-model fit peaked at 37 GB and "
            "the cardiac simulation at 24 GB in testing.\n\n"
            "Downloads: 'Set up MONAI-Physio in Slicer...' asks first, then "
            "installs PyTorch, torch-scatter and MONAI-Physio into Slicer's "
            "Python (several GB). The first run of each simulation also "
            "downloads its pretrained network (into "
            "~/.monai_physio/network_weights, after asking), and NV-Segment-CTMR "
            "(about 870 MB, from Hugging Face) when it is used, so it is much "
            "slower than later runs. 'Download sample' fetches an 89 MB chest "
            "CT. Nothing else is downloaded or sent."
        )
        parent.acknowledgementText = "Built on MONAI-Physio, 3D Slicer and OpenUSD."
        slicer.app.connect("startupCompleted()", register_sample_data)


class MONAIPhysioRespiratorySimulationLogic(ScriptedLoadableModuleLogic):
    """Worker command line and loading of the results into the scene."""

    @staticmethod
    def weights_directory() -> Path:
        """Where the pretrained networks are downloaded once and reused.

        The worker runs in this folder's parent, so registration weights that
        MONAI-Physio fetches into ./network_weights land here too. Kept in the
        home folder, like TotalSegmentator's ~/.totalsegmentator, because
        Slicer's cache folder can be cleared from its settings.
        """
        return Path.home() / ".monai_physio" / "network_weights"

    @staticmethod
    def weights_present(weights_dir: Path, simulation: str = "respiratory") -> bool:
        return (
            weights_dir / SIMULATIONS[simulation]["weights"] / "mgn_stage_model.pt"
        ).exists()

    @staticmethod
    def anatomy_color(simulation: str = "respiratory") -> tuple[float, float, float]:
        """Organ color from Slicer's GenericAnatomyColors table."""
        spec = SIMULATIONS[simulation]
        table = slicer.mrmlScene.GetFirstNodeByName("GenericAnatomyColors")
        if table is not None:
            index = table.GetColorIndexByName(spec["color_name"])
            rgba = [0.0, 0.0, 0.0, 0.0]
            if index >= 0 and table.GetColor(index, rgba):
                return (rgba[0], rgba[1], rgba[2])
        return spec["color"]

    @staticmethod
    def lung_color() -> tuple[float, float, float]:
        """Lung color from Slicer's GenericAnatomyColors table."""
        return MONAIPhysioRespiratorySimulationLogic.anatomy_color("respiratory")

    @staticmethod
    def write_volume(volume: Any, path: Path) -> None:
        """Write a volume for the worker without changing its storage node."""
        storage = slicer.vtkMRMLVolumeArchetypeStorageNode()
        storage.SetFileName(str(path))
        storage.SetUseCompression(False)
        if not storage.WriteData(volume):
            raise OSError(f"Could not write the CT to {path}")

    @staticmethod
    def worker_arguments(
        image: Path,
        name: str,
        weights_dir: Path,
        vtp_dir: Path,
        ct_dir: Optional[Path] = None,
        usd_dir: Optional[Path] = None,
        usd_colors: str = "slicer",
        slicer_color: tuple[float, float, float] = LUNG_COLOR,
        frames: int = 10,
        cycles_per_minute: float = 12.0,
        usd_name: Optional[str] = None,
        simulation: str = "respiratory",
        heart_segmenter: str = "simpleware",
        simpleware_exe: Optional[str] = None,
    ) -> list[str]:
        worker_name = SIMULATIONS[simulation]["worker"]
        worker = (
            Path(__file__).resolve().parent.parent / "MONAIPhysioExportUSD"
            / "MONAIPhysioLib" / worker_name
        )
        if not worker.is_file():
            import MONAIPhysioLib

            worker = Path(MONAIPhysioLib.__file__).resolve().parent / worker_name
        if not worker.is_file():
            raise FileNotFoundError(f"Bundled simulation worker is missing: {worker}")
        arguments = [
            "-u", str(worker), "--image", str(image), "--name", name,
            "--weights-dir", str(weights_dir), "--vtp-dir", str(vtp_dir),
        ]
        if ct_dir is not None:
            arguments.extend(["--ct-dir", str(ct_dir)])
        if usd_dir is not None:
            arguments.extend(["--usd-dir", str(usd_dir), "--usd-colors", usd_colors])
            arguments.extend(["--slicer-color", *(f"{c:.6g}" for c in slicer_color)])
            if usd_name:
                arguments.extend(["--usd-name", usd_name])
        arguments.extend([
            "--frames", str(frames), "--cycles-per-minute", f"{cycles_per_minute:g}",
        ])
        if simulation == "cardiac":
            arguments.extend(["--segmenter", heart_segmenter])
            if heart_segmenter == "simpleware" and simpleware_exe:
                arguments.extend(["--simpleware-exe", simpleware_exe])
        return arguments

    @staticmethod
    def output_directories(
        output: Path, surface_folder: str = "VTP", ct_folder: str = "CT"
    ) -> tuple[Path, Path, Path]:
        """(surface, 4D-CT, USD) folders: two subfolders of output, USD in it."""
        for label, folder in (("surface", surface_folder), ("4D-CT", ct_folder)):
            if (
                not folder
                or folder in (".", "..")
                or any(c in folder for c in '\\/:*?"<>|')
            ):
                raise ValueError(
                    f"The {label} subfolder name must be one folder name (not a path) "
                    f"inside the output folder, not '{folder}'."
                )
        return output / surface_folder, output / ct_folder, output

    @staticmethod
    def usd_file_name(text: str, default_stem: str, suffix: str = "respiration") -> str:
        """Validated USD file name; its stem names the root prim /World/<stem>."""
        name = text.strip() or f"{default_stem}_{suffix}.usd"
        if Path(name).suffix.lower() not in (".usd", ".usda", ".usdc"):
            name += ".usd"
        stem = Path(name).stem
        if Path(name).name != name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stem):
            raise ValueError(
                f"USD file name '{name}': use letters, digits and underscores, "
                "not starting with a digit (the name also names the USD root "
                "prim), ending in .usd, .usda or .usdc."
            )
        return name

    @staticmethod
    def _discard(node: Any) -> None:
        """Remove a node together with its display and storage nodes."""
        scene = slicer.mrmlScene
        for index in range(node.GetNumberOfDisplayNodes()):
            display = node.GetNthDisplayNode(index)
            if display is not None:
                scene.RemoveNode(display)
        storage = node.GetStorageNode() if hasattr(node, "GetStorageNode") else None
        if storage is not None:
            scene.RemoveNode(storage)
        scene.RemoveNode(node)

    def load_results(
        self, result: dict[str, Any],
        color: tuple[float, float, float] = LUNG_COLOR,
        simulation: str = "respiratory",
    ) -> list[Any]:
        """Build a surface sequence, and a CT sequence if one was written,
        under one browser. Returns the created nodes so a rerun can replace
        them."""
        spec = SIMULATIONS[simulation]
        scene = slicer.mrmlScene
        stages = result["stages"]
        surfaces = scene.AddNewNodeByClass("vtkMRMLSequenceNode", spec["surfaces"])
        surfaces.SetIndexName(spec["stage"])
        surfaces.SetIndexUnit("")
        for stage, path in zip(stages, result["surfaces"]):
            model = slicer.util.loadModel(path)
            surfaces.SetDataNodeAtValue(model, f"{stage:g}")
            self._discard(model)
        browser = scene.AddNewNodeByClass(
            "vtkMRMLSequenceBrowserNode", spec["browser"]
        )
        browser.SetAndObserveMasterSequenceNodeID(surfaces.GetID())
        # Play back at the simulated rate, cycle after cycle.
        if result.get("frames_per_second"):
            browser.SetPlaybackRateFps(float(result["frames_per_second"]))
        browser.SetPlaybackLooped(True)
        surface_proxy = browser.GetProxyNode(surfaces)
        if surface_proxy is not None and surface_proxy.GetDisplayNode() is not None:
            surface_proxy.GetDisplayNode().SetColor(*color)
        created = [browser, surfaces, surface_proxy]
        if result.get("images"):
            images = scene.AddNewNodeByClass("vtkMRMLSequenceNode", spec["images"])
            images.SetIndexName(spec["stage"])
            images.SetIndexUnit("")
            for stage, path in zip(stages, result["images"]):
                volume = slicer.util.loadVolume(path)
                images.SetDataNodeAtValue(volume, f"{stage:g}")
                self._discard(volume)
            browser.AddSynchronizedSequenceNodeID(images.GetID())
            image_proxy = browser.GetProxyNode(images)
            created += [images, image_proxy]
            if slicer.app.layoutManager() is not None:
                slicer.util.setSliceViewerLayers(background=image_proxy, fit=True)
        if slicer.app.layoutManager() is not None:
            slicer.util.resetThreeDViews()
        return created


class MONAIPhysioRespiratorySimulationWidget(ScriptedLoadableModuleWidget):
    """Run the worker asynchronously, then load its results."""

    def setup(self) -> None:
        super().setup()
        self.logic = MONAIPhysioRespiratorySimulationLogic()
        self.helper = MONAIPhysioHelper()
        self.helper.done_message = "Done. Results are loaded as sequences."
        self.helper.on_finished = self._run_finished
        self.created: list[Any] = []
        self.work: Optional[tempfile.TemporaryDirectory[str]] = None
        self.vtp_dir: Optional[Path] = None
        self.color = LUNG_COLOR
        self.run_simulation = "respiratory"
        self.helper.build_header(self.layout, with_models=False)

        form = self.helper.section(self.layout, "Input")
        self.simulation = qt.QComboBox()
        for key, spec in SIMULATIONS.items():
            self.simulation.addItem(spec["label"], key)
        self.simulation.toolTip = (
            "Respiratory: lung motion over a breathing cycle. Cardiac: heart "
            "motion over a heartbeat."
        )
        self.simulation.currentIndex = max(
            0, self.simulation.findData(self._setting("simulation", "respiratory"))
        )
        form.addRow("Simulation", self.simulation)
        self.volume = slicer.qMRMLNodeComboBox()
        self.volume.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volume.addEnabled = False
        self.volume.removeEnabled = False
        self.volume.noneEnabled = False
        self.volume.toolTip = (
            "Chest CT, loaded in Slicer, whose lungs or heart are simulated. "
            "NV-Segment-CTMR, used for the lungs and optionally for the heart, "
            "needs a CUDA GPU, downloads its weights once (about 870 MB) and is "
            "licensed for non-commercial research use only."
        )
        self.volume.setMRMLScene(slicer.mrmlScene)
        download = qt.QPushButton("Download sample")
        download.toolTip = (
            "Download and load the sample chest CT (89 MB, cached after the "
            "first download), the CT used by MONAI-Physio's tutorial 0"
        )
        download.clicked.connect(self._download_sample)
        row = qt.QHBoxLayout()
        row.addWidget(self.volume, 1)
        row.addWidget(download)
        form.addRow("Chest CT", row)

        self.heart_segmenter = qt.QComboBox()
        for key, label in HEART_SEGMENTERS:
            self.heart_segmenter.addItem(label, key)
        self.heart_segmenter.toolTip = (
            "The heart shape model and network were built from Simpleware "
            "Medical ASCardio segmentations, so Simpleware gives the geometry "
            "they expect (needs a Simpleware Medical installation). "
            "NV-Segment-CTMR is free for research use and needs a CUDA GPU, "
            "but its heart labels only approximate ASCardio's and on "
            "non-contrast CTs it can return the heart as one oversized region."
        )
        self.heart_segmenter.currentIndex = max(
            0, self.heart_segmenter.findData(self._setting("heartSegmenter", "simpleware"))
        )
        self.heart_segmenter_label = qt.QLabel("Heart segmenter")
        form.addRow(self.heart_segmenter_label, self.heart_segmenter)
        self.simpleware_exe = qt.QLineEdit()
        self.simpleware_exe.text = self._setting("simplewareExecutable", "")
        self.simpleware_exe.placeholderText = (
            "C:/Program Files/Synopsys/Simpleware Medical/Y-2026.03/"
            "ConsoleSimplewareMedical.exe"
        )
        self.simpleware_exe.toolTip = (
            "ConsoleSimplewareMedical.exe of the Simpleware Medical installation "
            "(with the ASCardio module). Leave empty for the default path shown."
        )
        browse_exe = qt.QToolButton()
        browse_exe.text = "..."
        browse_exe.toolTip = "Choose ConsoleSimplewareMedical.exe"
        browse_exe.clicked.connect(self._choose_simpleware)
        self.simpleware_row = qt.QWidget()
        row = qt.QHBoxLayout(self.simpleware_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.simpleware_exe)
        row.addWidget(browse_exe)
        self.simpleware_label = qt.QLabel("Simpleware executable")
        self.simpleware_label.indent = 20
        form.addRow(self.simpleware_label, self.simpleware_row)

        self.frames = qt.QSpinBox()
        self.frames.setRange(2, 50)
        self.frames.value = int(self._setting("frames", "10"))
        self.frames.toolTip = (
            "Stages simulated over one breathing or cardiac cycle (at most 50: "
            "MONAI-Physio names stage files in 0.01 steps)"
        )
        form.addRow("Frames per cycle", self.frames)
        self.cycles = qt.QDoubleSpinBox()
        self.cycles.setRange(1.0, 240.0)
        self.cycles.decimals = 1
        self.cycles.toolTip = (
            "Breathing or heart rate; sets the USD and sequence playback speed"
        )
        self.cycles_label = qt.QLabel()
        form.addRow(self.cycles_label, self.cycles)
        self.fps_label = qt.QLabel()
        form.addRow("Playback", self.fps_label)
        self.frames.valueChanged.connect(self._update_fps)
        self.cycles.valueChanged.connect(self._update_fps)

        form = self.helper.section(self.layout, "Output")
        self.output = qt.QLineEdit()
        self.output.text = self._setting("outputFolder", str(DEFAULT_FOLDER))
        self.output.toolTip = (
            "The surface and 4D-CT folders are created inside this folder; "
            "the USD file is written directly in it."
        )
        browse = qt.QToolButton()
        browse.text = "..."
        browse.toolTip = "Choose the output folder"
        browse.clicked.connect(self._choose_output)
        row = qt.QHBoxLayout()
        row.addWidget(self.output)
        row.addWidget(browse)
        form.addRow("Output folder", row)
        self.surface_folder = qt.QLineEdit()
        self.surface_folder.text = self._setting("surfaceSubfolder", "VTP")
        self.surface_folder.toolTip = (
            "One folder name, not a path: created inside the output folder for "
            "the organ surfaces"
        )
        form.addRow("VTK surface subfolder name", self.surface_folder)
        self.usd_enabled = qt.QCheckBox("Write an animated USD file")
        self.usd_enabled.checked = self._setting("usdEnabled", "false") == "true"
        form.addRow(self.usd_enabled)
        self.usd_name = qt.QLineEdit()
        self.usd_name.text = self._setting("usdName", "")
        self.usd_name.toolTip = (
            "File name in the output folder (.usd, .usda or .usdc). Letters, "
            "digits and underscores: it also names the USD root prim."
        )
        self._sub_row(form, "USD file name", self.usd_name, self.usd_enabled)
        self.usd_colors = qt.QComboBox()
        self.usd_colors.addItem("Slicer colors", "slicer")
        self.usd_colors.addItem("MONAI-Physio anatomy materials", "anatomy")
        self.usd_colors.toolTip = (
            "Slicer: the organ color from Slicer's GenericAnatomyColors table. "
            "Anatomy: MONAI-Physio's lung or heart OmniSurface material."
        )
        self.usd_colors.currentIndex = max(
            0, self.usd_colors.findData(self._setting("usdColors", "slicer"))
        )
        self._sub_row(form, "USD colors and materials", self.usd_colors, self.usd_enabled)
        self.ct_enabled = qt.QCheckBox("Write the 4D-CT (the CT warped to each stage)")
        self.ct_enabled.checked = self._setting("ctEnabled", "true") == "true"
        form.addRow(self.ct_enabled)
        self.ct_folder = qt.QLineEdit()
        self.ct_folder.text = self._setting("ctSubfolder", "CT")
        self.ct_folder.toolTip = (
            "One folder name, not a path: created inside the output folder for "
            "the warped CTs"
        )
        self._sub_row(form, "4D-CT subfolder name", self.ct_folder, self.ct_enabled)

        self.helper.build_footer(self.layout)
        self.helper.apply_button.clicked.connect(self._run)

        self.shown_simulation = self._simulation()
        self.cycles.value = self._rate_setting(self.shown_simulation)
        self.simulation.currentIndexChanged.connect(self._simulation_changed)
        self.heart_segmenter.currentIndexChanged.connect(self._update_simulation_ui)
        self.frames.valueChanged.connect(self._update_simulation_ui)
        self._update_simulation_ui()
        self._update_fps()

    @staticmethod
    def _sub_row(form: Any, label: str, field: Any, checkbox: Any) -> None:
        """Add a row indented under its checkbox, enabled only while checked."""
        text = qt.QLabel(label)
        text.indent = 20
        form.addRow(text, field)
        for widget in (text, field):
            checkbox.toggled.connect(widget.setEnabled)
            widget.enabled = checkbox.checked

    @staticmethod
    def _setting(key: str, default: str) -> str:
        return str(qt.QSettings().value(SETTINGS + key, default))

    def _simulation(self) -> str:
        return self.simulation.itemData(self.simulation.currentIndex) or "respiratory"

    def _heart_segmenter(self) -> str:
        return (
            self.heart_segmenter.itemData(self.heart_segmenter.currentIndex)
            or "simpleware"
        )

    def _rate_setting(self, simulation: str) -> float:
        spec = SIMULATIONS[simulation]
        return float(self._setting(spec["rate_setting"], f"{spec['default_rate']:g}"))

    def _simulation_changed(self, *args: Any) -> None:
        """Keep each simulation's own rate when switching between them."""
        previous = SIMULATIONS[self.shown_simulation]
        qt.QSettings().setValue(
            SETTINGS + previous["rate_setting"], f"{self.cycles.value:g}"
        )
        self.shown_simulation = self._simulation()
        self.cycles.value = self._rate_setting(self.shown_simulation)
        self._update_simulation_ui()

    def _update_simulation_ui(self, *args: Any) -> None:
        spec = SIMULATIONS[self._simulation()]
        cardiac = self._simulation() == "cardiac"
        for widget in (self.heart_segmenter_label, self.heart_segmenter):
            widget.visible = cardiac
        simpleware = cardiac and self._heart_segmenter() == "simpleware"
        for widget in (self.simpleware_label, self.simpleware_row):
            widget.visible = simpleware
        self.cycles_label.text = spec["rate_label"]
        self.usd_name.placeholderText = f"<CT name>_{spec['usd_suffix']}.usd"
        self.helper.apply_button.toolTip = (
            f"Segment, fit and predict {self.frames.value} "
            f"{spec['stage']}s, write the outputs and load them as sequences"
        )
        self._update_fps()

    def _download_sample(self) -> None:
        """Download (or reuse the cached) sample chest CT and select it."""
        import SampleData

        try:
            with slicer.util.WaitCursor():
                nodes = SampleData.SampleDataLogic().downloadFromURL(
                    uris=CHEST_CT["uris"],
                    fileNames=CHEST_CT["fileNames"],
                    nodeNames=CHEST_CT["nodeNames"],
                    checksums=CHEST_CT["checksums"],
                    loadFileTypes=CHEST_CT["loadFileType"],
                )
        except Exception as exc:
            slicer.util.errorDisplay(f"Could not download the sample CT: {exc}")
            return
        volumes = [n for n in nodes or [] if n.IsA("vtkMRMLScalarVolumeNode")]
        if volumes:
            self.volume.setCurrentNode(volumes[0])

    def _fps(self) -> float:
        return self.frames.value * self.cycles.value / 60.0

    def _update_fps(self, *args: Any) -> None:
        self.fps_label.text = (
            f"{self._fps():.3g} frames per second "
            f"({60.0 / self.cycles.value:.3g} s per cycle)"
        )

    def _choose_output(self) -> None:
        path = qt.QFileDialog.getExistingDirectory(
            self.parent, "Output folder", self.output.text
        )
        if path:
            self.output.text = path

    def _choose_simpleware(self) -> None:
        path = qt.QFileDialog.getOpenFileName(
            self.parent, "ConsoleSimplewareMedical.exe",
            self.simpleware_exe.text or self.simpleware_exe.placeholderText,
            "Executables (*.exe);;All files (*)",
        )
        if path:
            self.simpleware_exe.text = path

    def _save_settings(self) -> None:
        settings = qt.QSettings()
        for key, value in (
            ("usdEnabled", self.usd_enabled.checked),
            ("ctEnabled", self.ct_enabled.checked),
        ):
            settings.setValue(SETTINGS + key, "true" if value else "false")
        settings.setValue(SETTINGS + "simulation", self._simulation())
        settings.setValue(SETTINGS + "heartSegmenter", self._heart_segmenter())
        settings.setValue(SETTINGS + "simplewareExecutable", self.simpleware_exe.text.strip())
        settings.setValue(SETTINGS + "usdColors", self._usd_color_mode())
        settings.setValue(SETTINGS + "outputFolder", self.output.text.strip())
        settings.setValue(SETTINGS + "surfaceSubfolder", self.surface_folder.text.strip())
        settings.setValue(SETTINGS + "ctSubfolder", self.ct_folder.text.strip())
        settings.setValue(SETTINGS + "usdName", self.usd_name.text.strip())
        settings.setValue(SETTINGS + "frames", str(self.frames.value))
        settings.setValue(
            SETTINGS + SIMULATIONS[self._simulation()]["rate_setting"],
            f"{self.cycles.value:g}",
        )

    def _usd_color_mode(self) -> str:
        return self.usd_colors.itemData(self.usd_colors.currentIndex) or "slicer"

    def _run(self) -> None:
        if self.helper.process is not None:
            return
        try:
            simulation = self._simulation()
            spec = SIMULATIONS[simulation]
            volume = self.volume.currentNode()
            if volume is None:
                raise ValueError("Select a chest CT loaded in Slicer.")
            if not self.output.text.strip():
                raise ValueError("Choose the output folder.")
            heart_segmenter = self._heart_segmenter()
            simpleware_exe = self.simpleware_exe.text.strip()
            if (
                simulation == "cardiac" and heart_segmenter == "simpleware"
                and simpleware_exe and not Path(simpleware_exe).is_file()
            ):
                raise ValueError(
                    f"Simpleware Medical executable not found: {simpleware_exe}"
                )
            vtp_dir, ct_dir, usd_dir = self.logic.output_directories(
                Path(self.output.text.strip()),
                self.surface_folder.text.strip(),
                # Only checked when the 4D-CT is written.
                self.ct_folder.text.strip() if self.ct_enabled.checked else "CT",
            )
            name = re.sub(r"[^A-Za-z0-9_-]+", "_", volume.GetName()).strip("_") or "CT"
            usd_name = (
                self.logic.usd_file_name(
                    self.usd_name.text, name.replace("-", "_"), spec["usd_suffix"]
                )
                if self.usd_enabled.checked else None
            )
            weights_dir = self.logic.weights_directory()
            if not self.logic.weights_present(weights_dir, simulation) and (
                not slicer.util.confirmOkCancelDisplay(
                    f"The pretrained {spec['network']} network is downloaded "
                    f"once into {weights_dir}. Continue?",
                    windowTitle="MONAI-Physio Simulation",
                )
            ):
                return
            self._save_settings()
            self.helper.log.clear()
            if volume.GetParentTransformNode() is not None:
                self.helper.log.appendPlainText(
                    "Note: the CT's parent transform is not applied; harden it "
                    "first to simulate in the transformed space."
                )
            self.work = tempfile.TemporaryDirectory(prefix=f"slicer-{simulation}-")
            image = Path(self.work.name) / "ct.nrrd"
            self.logic.write_volume(volume, image)
            self.color = self.logic.anatomy_color(simulation)
            arguments = self.logic.worker_arguments(
                image, name, weights_dir, vtp_dir,
                ct_dir if self.ct_enabled.checked else None,
                usd_dir if self.usd_enabled.checked else None,
                self._usd_color_mode(), self.color,
                self.frames.value, self.cycles.value, usd_name,
                simulation, heart_segmenter, simpleware_exe or None,
            )
            self.vtp_dir = vtp_dir
            self.run_simulation = simulation
            self.helper.log.appendPlainText(
                f"Simulating {simulation} motion for '{volume.GetName()}'"
            )
            # The shape-model fit downloads registration weights into
            # ./network_weights; run in the cache so they land beside the
            # pretrained network instead of wherever Slicer was started.
            self.helper.start_worker(
                python_slicer(), arguments, weights_dir.parent
            )
        except Exception as exc:
            self._cleanup_work()
            self.helper._release()
            slicer.util.errorDisplay(str(exc))

    def _cleanup_work(self) -> None:
        if self.work is not None:
            self.work.cleanup()
            self.work = None

    def _run_finished(self, success: bool) -> None:
        self._cleanup_work()
        if not success or self.vtp_dir is None:
            return
        try:
            result_file = self.vtp_dir / "result.json"
            result = json.loads(result_file.read_text(encoding="utf-8"))
            for node in self.created:
                if node is not None and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
            self.created = self.logic.load_results(
                result, self.color, self.run_simulation
            )
            self.helper.log.appendPlainText(
                f"Loaded {len(result['surfaces'])} surfaces"
                + (f" and {len(result['images'])} CT volumes" if result["images"] else "")
                + " as sequences."
                + (f" USD: {result['usd']}" if result.get("usd") else "")
            )
        except Exception as exc:
            self.helper.log.appendPlainText(f"Could not load results: {exc}")
            slicer.util.errorDisplay(f"Could not load results: {exc}")

    def cleanup(self) -> None:
        self.helper.cleanup()
        self._cleanup_work()


class MONAIPhysioRespiratorySimulationTest(ScriptedLoadableModuleTest):
    """Check the worker command line for each output combination."""

    def runTest(self) -> None:
        self.test_arguments()
        self.test_cardiac_arguments()

    def test_arguments(self) -> None:
        logic = MONAIPhysioRespiratorySimulationLogic
        minimal = logic.worker_arguments(
            Path("ct.nrrd"), "CT", Path("w"), Path("vtp")
        )
        self.assertTrue(Path(minimal[1]).is_file())
        for flag in ("--ct-dir", "--usd-dir", "--segmenter"):
            self.assertNotIn(flag, minimal)
        vtp, ct, usd = logic.output_directories(Path("out"))
        self.assertEqual((vtp, ct, usd), (Path("out/VTP"), Path("out/CT"), Path("out")))
        self.assertEqual(
            logic.output_directories(Path("out"), "Meshes", "CT4D")[:2],
            (Path("out/Meshes"), Path("out/CT4D")),
        )
        for bad in ("", "..", "a/b", "a\\b"):
            with self.assertRaises(ValueError):
                logic.output_directories(Path("out"), bad)
        self.assertEqual(logic.usd_file_name("", "Chest_CT"), "Chest_CT_respiration.usd")
        self.assertEqual(logic.usd_file_name("Lungs", "x"), "Lungs.usd")
        self.assertEqual(logic.usd_file_name("Lungs.usdc", "x"), "Lungs.usdc")
        for bad in ("2lungs.usd", "my lungs.usd", "a/b.usd"):
            with self.assertRaises(ValueError):
                logic.usd_file_name(bad, "x")
        full = logic.worker_arguments(
            Path("ct.nrrd"), "CT", Path("w"), vtp, ct, usd, "anatomy",
            (0.5, 0.5, 0.5),
        )
        for flag in ("--ct-dir", "--usd-dir", "--usd-colors"):
            self.assertIn(flag, full)
        self.assertEqual(full[full.index("--usd-colors") + 1], "anatomy")

    def test_cardiac_arguments(self) -> None:
        logic = MONAIPhysioRespiratorySimulationLogic
        self.assertEqual(
            logic.usd_file_name("", "Chest_CT", "heartbeat"), "Chest_CT_heartbeat.usd"
        )
        simpleware = logic.worker_arguments(
            Path("ct.nrrd"), "CT", Path("w"), Path("vtp"),
            simulation="cardiac", simpleware_exe="C:/sw/Console.exe",
        )
        self.assertTrue(Path(simpleware[1]).is_file())
        self.assertEqual(Path(simpleware[1]).name, "cardiac_simulation_worker.py")
        self.assertEqual(simpleware[simpleware.index("--segmenter") + 1], "simpleware")
        self.assertEqual(
            simpleware[simpleware.index("--simpleware-exe") + 1], "C:/sw/Console.exe"
        )
        nvsegment = logic.worker_arguments(
            Path("ct.nrrd"), "CT", Path("w"), Path("vtp"),
            simulation="cardiac", heart_segmenter="nvsegment",
            simpleware_exe="C:/sw/Console.exe",
        )
        self.assertEqual(nvsegment[nvsegment.index("--segmenter") + 1], "nvsegment")
        self.assertNotIn("--simpleware-exe", nvsegment)
