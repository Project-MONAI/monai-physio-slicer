# SPDX-License-Identifier: Apache-2.0
"""Helper class and logic shared by the MONAI Physio Slicer modules."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import qt
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

from MONAIPhysioLib.snapshot import capture_model
from MONAIPhysioLib.runtime import (
    ensure_native_dependencies,
    installed_torch,
    monai_physio_problem,
    python_slicer,
    torch_choices,
)

# Workers print "PROGRESS <percent> <message>" to drive the progress bar.
PROGRESS_LINE = re.compile(r"^PROGRESS (\d{1,3})(?: (.*))?$")


class MONAIPhysioLogic(ScriptedLoadableModuleLogic):
    """Model discovery, capture and worker invocation independent of the GUI."""

    @staticmethod
    def visible_models() -> list[Any]:
        """Visible, non-hidden model nodes with a surface to export."""
        models = []
        for model in slicer.util.getNodesByClass("vtkMRMLModelNode"):
            display = model.GetDisplayNode()
            if (
                model.GetHideFromEditors()
                or display is None
                or not (display.GetVisibility() and display.GetVisibility3D())
                or model.GetPolyData() is None
                or not model.GetPolyData().GetNumberOfPoints()
            ):
                continue
            models.append(model)
        return models

    def capture(self, model: Any, directory: Path) -> Path:
        """Capture one model's geometry and supported appearance without edits."""
        return capture_model(model, directory)

    @staticmethod
    def worker_arguments(
        snapshots: Any,
        output: Path,
        overwrite: bool = False,
        colors: str = "slicer",
        mesh_type: str = "surface",
        resolution: float = 2.0,
    ) -> list[str]:
        """Return argv for the bundled worker, usable from source or install."""
        worker = Path(__file__).resolve().parent / "export_worker.py"
        if not worker.is_file():
            raise FileNotFoundError(f"Bundled export worker is missing: {worker}")
        if isinstance(snapshots, (str, Path)):
            snapshots = [snapshots]
        arguments = ["-u", str(worker), str(output), *map(str, snapshots)]
        if overwrite:
            arguments.append("--overwrite")
        if colors != "slicer":
            arguments.extend(["--colors", colors])
        if mesh_type != "surface":
            arguments.extend(
                ["--mesh", mesh_type, "--resolution", f"{resolution:.6g}"]
            )
        return arguments


class MONAIPhysioHelper:
    """Common widget parts: setup button, model choice, worker process, log.

    The layout follows Slicer's CLI modules (e.g. Model Maker): parameter
    sections in collapsible boxes, then a progress bar and a bottom row with
    an optional AutoRun menu button, Cancel and Apply. A module widget builds
    its sections between build_header() and build_footer(), connects
    apply_button, and calls run() or start_worker(). on_finished, if set, is
    called with a success flag after each run ends (used to chain AutoRun).
    """

    AUTO_RUN_OPTIONS = (
        ("on_parameter", "AutoRun on changed parameter",
         "As long as the AutoRun button is down, the module is run anytime a "
         "parameter value is changed.", True),
        ("on_input", "AutoRun on modified input",
         "As long as the AutoRun button is down, the module is run anytime an "
         "input model, its display or its transform is modified, or models "
         "are added or removed.", True),
        ("cancels", "AutoRun cancels running process",
         "When checked, a new AutoRun stops the running export; otherwise it "
         "waits for the running export to finish.", False),
    )

    def __init__(self) -> None:
        self.logic = MONAIPhysioLogic()
        self.process: Optional[Any] = None
        self.work: Optional[tempfile.TemporaryDirectory[str]] = None
        self.action_buttons: list[Any] = []
        self.on_finished: Optional[Any] = None
        # "slicer" display colors or "anatomy" MONAI-Physio materials.
        self.colors = "slicer"
        # "surface", or "tetrahedra"/"hexahedra" of resolution millimeters.
        self.mesh_type = "surface"
        self.resolution = 2.0
        self.done_message = "Done. Open or reference the USD asset in Omniverse."
        self._partial = ""  # incomplete last line of worker output
        self._finished_success = False
        self._notify_timer = qt.QTimer()
        self._notify_timer.setSingleShot(True)
        self._notify_timer.timeout.connect(self._notify_finished)

    def build_header(self, layout: Any, with_models: bool = True) -> None:
        """PyTorch choice and setup button first, a large gap, then the
        one-or-all model choice."""
        torch_row = qt.QFormLayout()
        layout.addLayout(torch_row)
        self.torch_build = qt.QComboBox()
        self.torch_build.toolTip = (
            "PyTorch build that setup installs. Keep the installed build, or "
            "pick another (it is force-reinstalled, then Slicer must restart). "
            "RTX 50 / Blackwell GPUs need CUDA 12.8 or newer."
        )
        current = installed_torch()
        self.torch_build.addItem(
            f"{current} (installed)" if current else "Not installed (pip default)",
            current or "",
        )
        for label, build in torch_choices():
            if build != current:
                self.torch_build.addItem(label, build)
        torch_row.addRow("PyTorch", self.torch_build)
        self.setup_button = qt.QPushButton("Set up MONAI-Physio in Slicer...")
        self.setup_button.clicked.connect(self._setup_dependencies)
        layout.addWidget(self.setup_button)
        layout.addSpacing(60)
        if not with_models:
            return
        models = self.section(layout, "Models")
        self.export_all = qt.QCheckBox("Export all visible surface models")
        self.export_all.checked = True
        # A row layout, so modules can add widgets after the checkbox.
        self.export_all_row = qt.QHBoxLayout()
        self.export_all_row.addWidget(self.export_all)
        self.export_all_row.addStretch(1)
        models.addRow(self.export_all_row)
        self.model = slicer.qMRMLNodeComboBox()
        self.model.nodeTypes = ["vtkMRMLModelNode"]
        self.model.addEnabled = False
        self.model.removeEnabled = False
        self.model.setMRMLScene(slicer.mrmlScene)
        self.model.enabled = False
        self.export_all.toggled.connect(lambda all_: self.model.setEnabled(not all_))
        models.addRow("Single model", self.model)

    @staticmethod
    def section(layout: Any, title: str) -> Any:
        """Add a collapsible parameter section and return its form layout."""
        import ctk

        box = ctk.ctkCollapsibleButton()
        box.text = title
        layout.addWidget(box)
        return qt.QFormLayout(box)

    def build_footer(self, layout: Any, auto_run: bool = False) -> None:
        """Log section, then progress bar and [AutoRun] ... [Cancel] [Apply]."""
        log_form = self.section(layout, "Log")
        self.log = qt.QPlainTextEdit()
        self.log.readOnly = True
        log_form.addRow(self.log)
        layout.addStretch(1)
        self.progress = qt.QProgressBar()
        self.progress.visible = False
        layout.addWidget(self.progress)
        row = qt.QHBoxLayout()
        layout.addLayout(row)
        self.auto_run_button = None
        self.auto_run_actions: dict[str, Any] = {}
        if auto_run:
            import ctk

            self.auto_run_button = ctk.ctkMenuButton()
            self.auto_run_button.text = "AutoRun"
            self.auto_run_button.checkable = True
            menu = qt.QMenu(self.auto_run_button)
            settings = qt.QSettings()
            for key, text, tip, default in self.AUTO_RUN_OPTIONS:
                action = qt.QAction(text, menu)
                action.toolTip = tip
                action.checkable = True
                action.checked = str(settings.value(
                    f"MONAIPhysio/autoRun/{key}", str(default).lower()
                )).lower() == "true"
                action.toggled.connect(
                    lambda on, key=key: qt.QSettings().setValue(
                        f"MONAIPhysio/autoRun/{key}", "true" if on else "false"
                    )
                )
                menu.addAction(action)
                self.auto_run_actions[key] = action
            menu.toolTipsVisible = True
            self.auto_run_button.setMenu(menu)
            row.addWidget(self.auto_run_button)
        row.addStretch(1)
        self.cancel_button = qt.QPushButton("Cancel")
        self.cancel_button.toolTip = "Cancel the running export"
        self.cancel_button.enabled = False
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.cancel_button)
        self.apply_button = qt.QPushButton("Apply")
        self.apply_button.toolTip = "Run the module"
        row.addWidget(self.apply_button)
        self.add_action_button(self.apply_button)

    def auto_run_option(self, key: str) -> bool:
        action = self.auto_run_actions.get(key)
        return bool(action is not None and action.checked)

    def add_action_button(self, button: Any) -> None:
        """Buttons disabled while a worker runs."""
        self.action_buttons.append(button)

    def _setup_dependencies(self) -> None:
        """Use Slicer's package manager on explicit request, never at startup."""
        if self.process is not None:
            return
        build = self.torch_build.itemData(self.torch_build.currentIndex) or None
        changing = bool(build) and build != installed_torch()
        try:
            if ensure_native_dependencies(build, self.log.appendPlainText):
                self.log.appendPlainText(
                    "Dependency setup completed. Restart Slicer before exporting "
                    "if packages were installed or changed."
                )
                if changing and slicer.util.confirmYesNoDisplay(
                    f"PyTorch {build} is installed. Restart Slicer now to use it?",
                    windowTitle="MONAI Physio setup",
                ):
                    slicer.util.restart()
        except Exception as exc:
            self.log.appendPlainText(str(exc))
            slicer.util.errorDisplay(
                f"MONAI-Physio setup did not complete: {exc}\n"
                "See the installation guide."
            )

    def run(self, output: Path, quiet: bool = False) -> None:
        """Capture the chosen models and convert them in the worker process.

        An existing output file is replaced. quiet logs problems instead of
        raising a dialog, for unattended auto-updates.
        """
        if self.process is not None:
            return
        try:
            python = python_slicer()
            if self.export_all.checked:
                models = self.logic.visible_models()
                if not models:
                    raise ValueError("No visible surface models to export.")
            else:
                models = [self.model.currentNode()]
            self.log.clear()
            self.work = tempfile.TemporaryDirectory(prefix="slicer-monai-physio-")
            work = Path(self.work.name)
            snapshots = []
            for index, model in enumerate(models):
                try:
                    snapshots.append(self.logic.capture(model, work / f"model{index}"))
                except Exception as exc:
                    if not self.export_all.checked:
                        raise
                    name = model.GetName() if model else "?"
                    self.log.appendPlainText(f"Skipped '{name}': {exc}")
            if not snapshots:
                raise ValueError("None of the visible models could be exported.")
            self.log.appendPlainText(
                f"Exporting {len(snapshots)} model(s). Phong lighting is "
                "approximated by USD Preview Surface. Geometry and world "
                "placement are preserved; transforms are baked."
            )
            arguments = self.logic.worker_arguments(
                snapshots, output, True, self.colors,
                self.mesh_type, self.resolution,
            )
            self.start_worker(python, arguments)
        except Exception as exc:
            self._release()
            if quiet:
                self.log.appendPlainText(str(exc))
            else:
                slicer.util.errorDisplay(str(exc))
            self._notify(False)

    def start_worker(
        self, python: Path, arguments: list[str],
        working_directory: Optional[Path] = None,
    ) -> None:
        """Start the worker process, streaming its output to the log.

        Some MONAI-Physio steps download into paths relative to the working
        directory, so long-running workers should be given a cache folder.
        Raises RuntimeError if the installed MONAI-Physio is missing or too old.
        """
        problem = monai_physio_problem()
        if problem:
            raise RuntimeError(problem)
        # Parented, so PythonQt never deletes it when self.process is cleared:
        # that happens inside its own finished signal, and deleting the
        # sender there crashes Slicer. deleteLater() in _release frees it.
        self.process = qt.QProcess(slicer.app)
        if working_directory is not None:
            working_directory.mkdir(parents=True, exist_ok=True)
            self.process.setWorkingDirectory(str(working_directory))
        environment = qt.QProcessEnvironment()
        for key, value in slicer.util.startupEnvironment().items():
            environment.insert(key, value)
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_log)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self._set_busy(True)
        self.process.start(str(python), arguments)

    def _notify_finished(self) -> None:
        self._notify(self._finished_success)

    def _notify(self, success: bool) -> None:
        if self.on_finished is not None:
            self.on_finished(success)

    def _set_busy(self, busy: bool) -> None:
        for button in self.action_buttons:
            button.enabled = not busy
        self.setup_button.enabled = not busy
        self.cancel_button.enabled = busy
        # Busy indicator until the worker reports progress, as Slicer's CLI
        # progress bar shows while running.
        self.progress.visible = busy
        self.progress.setRange(0, 0 if busy else 1)
        self.progress.format = "%p%"
        self._partial = ""

    def set_progress(self, percent: int, message: str = "") -> None:
        self.progress.setRange(0, 100)
        # Label first, so the value change repaints with the matching text.
        self.progress.format = f"{message} (%p%)" if message else "%p%"
        self.progress.value = max(0, min(100, percent))

    def _read_log(self) -> None:
        if self.process is not None:
            # PythonQt's QByteArray does not support bytes(); data() does.
            data = self.process.readAllStandardOutput().data()
            message = (
                data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
            )
            # Output arrives in arbitrary chunks; handle complete lines only.
            lines = (self._partial + message).split("\n")
            self._partial = lines.pop()
            for line in lines:
                self._handle_line(line.rstrip("\r"))

    def _flush_log(self) -> None:
        self._read_log()
        if self._partial:
            self._handle_line(self._partial.rstrip("\r"))
            self._partial = ""

    def _handle_line(self, line: str) -> None:
        """Worker "PROGRESS <percent> <message>" lines drive the progress bar."""
        match = PROGRESS_LINE.match(line)
        if match is None:
            self.log.appendPlainText(line)
            return
        percent, message = int(match.group(1)), match.group(2) or ""
        self.set_progress(percent, message)
        self.log.appendPlainText(f"[{percent}%] {message}")

    def _finished(self, code: int, status: Any = None) -> None:
        # PythonQt connects finished(int): the exit status is not passed.
        if status is None and self.process is not None:
            status = self.process.exitStatus()
        success = code == 0 and status == qt.QProcess.NormalExit
        try:
            self._flush_log()
            self.log.appendPlainText(
                self.done_message
                if success else "Run failed or was cancelled; see log above."
            )
        finally:
            self._release()
        # Run the callback after this signal handler returns. Callbacks that
        # load data pump the event loop, which would delete the QProcess
        # (deleteLater above) while Qt is still emitting its finished signal
        # and crash Slicer with an access violation.
        # A persistent single-shot timer connected to a method: a temporary
        # lambda passed to QTimer.singleShot is not kept alive by PythonQt
        # and can be freed before it fires (intermittent access violation).
        self._finished_success = success
        self._notify_timer.start(0)

    def _error(self, error: Any) -> None:
        if self.process is not None:
            self.log.appendPlainText(self.process.errorString())
            if error == qt.QProcess.FailedToStart:
                self._release()
                self._notify(False)

    def cancel(self) -> None:
        """Stop the running worker; finished/on_finished still fire."""
        if self.process is not None:
            self.process.kill()

    def _release(self) -> None:
        if self.process is not None:
            self.process.deleteLater()
            self.process = None
        if self.work is not None:
            self.work.cleanup()
            self.work = None
        self._set_busy(False)

    def cleanup(self) -> None:
        """Stop the worker before removing its snapshot files."""
        self.on_finished = None
        self._notify_timer.stop()
        process = self.process
        if process is not None:
            process.kill()
            process.waitForFinished(3000)
        self._release()
