# SPDX-License-Identifier: Apache-2.0
"""Validate both MONAI-Physio modules end to end inside a real 3D Slicer.

Drives the modules through their own widgets, as a user would, and writes a
results JSON. Run it in three Slicer sessions, each started with the module
paths and a phase selected through environment variables:

    set MONAI_PHYSIO_VALIDATION_PHASE=setup   (then test, then cpu)
    set MONAI_PHYSIO_VALIDATION_OUTPUT=C:\\path\\to\\results
    Slicer.exe --no-splash --disable-settings ^
        --additional-module-paths <repo>\\MONAIPhysioExportUSD ^
        <repo>\\MONAIPhysioRespiratorySimulation ^
        --python-script <repo>\\tools\\slicer_validation.py

Phases:
    setup  "Set up MONAI-Physio in Slicer..." with the PyTorch build in
           MONAI_PHYSIO_VALIDATION_TORCH (default 2.14.1+cu132), answering
           yes to its confirmations. Installs into Slicer's Python.
    test   Module registration, both self-tests, USD export of the sample
           models (surface and tetrahedra), and a respiratory simulation of
           the sample CT with the animated USD and the 4D-CT. Needs a CUDA
           GPU and downloads the sample data and network weights.
    cpu    Start Slicer with CUDA_VISIBLE_DEVICES=-1: surface and hexahedra
           export must succeed and tetrahedra export must fail with a
           reported error.

Not a unit test: it installs packages and downloads data. Use it on a Slicer
installation you can restore, and run it with --disable-settings so the
user's Slicer settings are not changed.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import qt
import slicer

# Slicer, Qt and module-widget objects are PythonQt wrappers without type
# stubs, so they are typed Any throughout; check results are free-form JSON.

PHASE = os.environ.get("MONAI_PHYSIO_VALIDATION_PHASE", "test")
OUTPUT = Path(os.environ.get(
    "MONAI_PHYSIO_VALIDATION_OUTPUT", Path.home() / "monai-physio-validation"
))
TORCH_BUILD = os.environ.get("MONAI_PHYSIO_VALIDATION_TORCH", "2.14.1+cu132")
WORKER_TIMEOUT_S = 3 * 60 * 60

MODULES = ("MONAIPhysioExportUSD", "MONAIPhysioRespiratorySimulation")
CATEGORY = "Simulation"


class Validation:
    """Collects check results, mirrors them to a log file and the console."""

    def __init__(self, output: Path, phase: str) -> None:
        self.output = output / phase
        self.output.mkdir(parents=True, exist_ok=True)
        self.log_file = (self.output / "validation.log").open("w", encoding="utf-8")
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.info: dict[str, Any] = {"phase": phase}

    def log(self, message: str) -> None:
        line = f"[validation] {message}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def check(self, name: str, function: Callable[[], Any]) -> Any:
        """Run one named check; record pass/fail, duration and any detail."""
        self.log(f"START {name}")
        start = time.time()
        try:
            detail = function()
            passed = True
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.log(traceback.format_exc())
            passed = False
        seconds = round(time.time() - start, 1)
        self.checks.append(
            {"name": name, "passed": passed, "seconds": seconds, "detail": detail}
        )
        self.log(f"{'PASS' if passed else 'FAIL'} {name} ({seconds} s): {detail}")
        return detail if passed else None

    def finish(self) -> int:
        passed = all(c["passed"] for c in self.checks) and bool(self.checks)
        result = {
            **self.info,
            "passed": passed,
            "checks": self.checks,
            "dialog_errors": self.errors,
            "slicer_error_log": slicer_error_log(),
        }
        (self.output / "results.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        self.log(f"VERDICT: {'PASS' if passed else 'FAIL'}")
        self.log_file.close()
        return 0 if passed else 1


def slicer_error_log() -> list[str]:
    """Warnings and errors in Slicer's error log (the status-bar icon)."""
    model = slicer.app.errorLogModel()
    columns = {
        str(model.headerData(c, qt.Qt.Horizontal)): c
        for c in range(model.columnCount())
    }
    level, description = columns.get("Level", 2), columns.get("Description", 4)
    entries = []
    for row in range(model.rowCount()):
        text = str(model.data(model.index(row, level)))
        if text in ("Error", "Critical", "Fatal", "Warning"):
            entries.append(f"{text}: {model.data(model.index(row, description))}")
    return entries


def wait_until(condition: Callable[[], bool], timeout_s: float) -> None:
    """Pump Slicer's event loop until condition() holds or time runs out."""
    deadline = time.time() + timeout_s
    while not condition():
        if time.time() > deadline:
            raise TimeoutError(f"not done after {timeout_s:.0f} s")
        slicer.app.processEvents()
        time.sleep(0.05)


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi(query: str) -> Optional[str]:
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip().splitlines()[0]
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "slicer": slicer.app.applicationVersion,
        "slicer_revision": slicer.app.revision,
        "python": platform.python_version(),
        "os": platform.platform(),
        "gpu": nvidia_smi("name,driver_version,memory.total"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for name in (
        "monai-physio", "torch", "torchvision", "torch-scatter",
        "torch-geometric", "nvidia-physicsnemo", "warp-lang", "usd-core",
        "pyvista", "numpy", "vtk", "monai",
    ):
        info[name] = package_version(name)
    return info


def patch_dialogs(validation: Validation) -> None:
    """Answer confirmations with yes and record error dialogs, unattended."""
    util = slicer.util
    util.confirmYesNoDisplay = lambda *a, **k: True
    util.confirmOkCancelDisplay = lambda *a, **k: True

    def record(text: Any, *args: Any, **kwargs: Any) -> None:
        validation.errors.append(str(text))
        validation.log(f"error dialog: {text}")

    util.errorDisplay = record
    util.warningDisplay = record
    # Setup offers a restart after changing PyTorch; the runner restarts.
    util.restart = lambda *a, **k: validation.log("restart requested (skipped)")


def module_widget(name: str) -> Any:
    slicer.util.selectModule(name)
    slicer.app.processEvents()
    return slicer.util.getModuleWidget(name)


def run_helper(helper: Any, start: Callable[[], None]) -> bool:
    """Start a worker through the module and wait for its finished callback."""
    outcome: dict[str, bool] = {}
    previous = helper.on_finished

    def finished(success: bool) -> None:
        outcome["success"] = success
        if previous is not None:
            previous(success)

    helper.on_finished = finished
    try:
        start()
        wait_until(lambda: "success" in outcome, WORKER_TIMEOUT_S)
        # Let the module's own callback (loading results) complete.
        for _ in range(20):
            slicer.app.processEvents()
    finally:
        helper.on_finished = previous
    return outcome["success"]


def screenshot(validation: Validation, name: str, layout: int) -> str:
    """Save the main window at a fixed size with the given view layout."""
    window = slicer.util.mainWindow()
    window.showNormal()
    window.resize(1600, 1000)
    probe = window.findChild("ctkCollapsibleButton", "DataProbeCollapsibleWidget")
    if probe is not None:
        probe.collapsed = True
    manager = slicer.app.layoutManager()
    manager.setLayout(layout)
    for _ in range(10):
        slicer.app.processEvents()
    view = manager.threeDWidget(0).threeDView()
    view.resetFocalPoint()
    view.resetCamera()
    slicer.util.resetSliceViews()
    for _ in range(10):
        slicer.app.processEvents()
        time.sleep(0.05)
    path = validation.output / f"{name}.png"
    window.grab().save(str(path))
    return str(path)


# ---------------------------------------------------------------- setup


def phase_setup(validation: Validation) -> None:
    validation.info["before"] = environment_info()

    def setup() -> str:
        widget = module_widget("MONAIPhysioExportUSD")
        helper = widget.helper
        index = helper.torch_build.findData(TORCH_BUILD)
        if index < 0:
            choices = [
                helper.torch_build.itemData(i) for i in range(helper.torch_build.count)
            ]
            raise ValueError(f"{TORCH_BUILD} not offered; choices: {choices}")
        helper.torch_build.currentIndex = index
        helper._setup_dependencies()
        log = helper.log.toPlainText()
        (validation.output / "setup-log.txt").write_text(log, encoding="utf-8")
        if "Dependency setup completed" not in log:
            raise RuntimeError("setup did not report completion; see setup-log.txt")
        return f"PyTorch {TORCH_BUILD} selected; setup completed"

    validation.check("setup through the module", setup)
    validation.info["after"] = environment_info()


# ----------------------------------------------------------------- test


def check_registration() -> str:
    import SampleData

    found = []
    for name in MODULES:
        module = slicer.app.moduleManager().module(name)
        if module is None:
            raise RuntimeError(f"{name} is not registered")
        if CATEGORY not in list(module.categories):
            raise RuntimeError(f"{name} categories {list(module.categories)}")
        found.append(name)
    sources = getattr(slicer.modules, "sampleDataSources", {})
    names = [s.sampleName for s in sources.get(CATEGORY, [])]
    for sample in ("MONAI-Physio-Sample-Models", "Chest-CT"):
        if sample not in names:
            raise RuntimeError(f"sample {sample} not in '{CATEGORY}': {names}")
    del SampleData
    from MONAIPhysioLib.runtime import monai_physio_problem

    problem = monai_physio_problem()
    if problem:
        raise RuntimeError(problem)
    return f"{found} under '{CATEGORY}', samples registered, MONAI-Physio usable"


def run_self_test(name: str) -> str:
    module = __import__(name)
    test = getattr(module, f"{name}Test")()
    test.runTest()
    return "runTest passed"


def check_usd(path: Path, minimum_meshes: int, prim_type: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    prims = [
        p for p in stage.Traverse()
        if p.GetPath().pathString.startswith("/World/SlicerModel/")
        and p.GetTypeName() == prim_type
    ]
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    if len(prims) < minimum_meshes:
        raise RuntimeError(
            f"{len(prims)} {prim_type} prims, expected {minimum_meshes}+"
        )
    if abs(meters - 1.0) > 1e-9 or up != UsdGeom.Tokens.y:
        raise RuntimeError(f"metersPerUnit {meters}, upAxis {up}")
    return {
        "file": str(path), "bytes": path.stat().st_size,
        prim_type: len(prims), "metersPerUnit": meters, "upAxis": str(up),
    }


def export_usd(
    validation: Validation, mesh_type: str, single: Optional[str] = None,
    resolution: float = 4.0,
) -> tuple[bool, Path, Any]:
    widget = module_widget("MONAIPhysioExportUSD")
    if not widget.helper.logic.visible_models():
        widget._download_sample()
    widget.mesh_type.currentIndex = widget.mesh_type.findData(mesh_type)
    widget.resolution.value = resolution
    widget.helper.export_all.checked = single is None
    if single is not None:
        widget.helper.model.setCurrentNode(slicer.util.getNode(single))
    target = validation.output / f"export-{mesh_type}.usdc"
    target.unlink(missing_ok=True)
    widget.output.text = str(target)
    success = run_helper(widget.helper, widget._apply)
    (validation.output / f"export-{mesh_type}-log.txt").write_text(
        widget.helper.log.toPlainText(), encoding="utf-8"
    )
    return success, target, widget


def first_lung_model() -> str:
    for model in slicer.util.getNodesByClass("vtkMRMLModelNode"):
        if "lung" in model.GetName().lower():
            return model.GetName()
    raise RuntimeError("no lung model in the sample scene")


def phase_test(validation: Validation) -> None:
    validation.info["environment"] = environment_info()
    validation.check("modules registered", check_registration)
    for name in MODULES:
        validation.check(f"{name} self-test", lambda name=name: run_self_test(name))

    def surface() -> dict[str, Any]:
        success, target, widget = export_usd(validation, "surface")
        if not success:
            raise RuntimeError("export failed; see export-surface-log.txt")
        count = len(widget.helper.logic.visible_models())
        return {**check_usd(target, count, "Mesh"), "visible_models": count}

    validation.check("USD export: all sample models, surface", surface)
    validation.check(
        "screenshot: USD export",
        lambda: screenshot(
            validation, "export-usd",
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView,
        ),
    )

    def tetrahedra() -> dict[str, Any]:
        model = first_lung_model()
        success, target, _ = export_usd(validation, "tetrahedra", single=model)
        if not success:
            raise RuntimeError("export failed; see export-tetrahedra-log.txt")
        return {**check_usd(target, 1, "TetMesh"), "model": model}

    validation.check("USD export: one model, tetrahedra 4 mm", tetrahedra)
    slicer.mrmlScene.Clear(0)
    validation.check("respiratory simulation", lambda: respiration(validation))
    validation.check(
        "screenshot: respiratory simulation",
        lambda: screenshot(
            validation, "respiratory-simulation",
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView,
        ),
    )


def respiration(validation: Validation) -> dict[str, Any]:
    widget = module_widget("MONAIPhysioRespiratorySimulation")
    widget._download_sample()
    if widget.volume.currentNode() is None:
        raise RuntimeError("sample CT did not load")
    folder = validation.output / "respiration"
    widget.output.text = str(folder)
    widget.usd_enabled.checked = True
    widget.ct_enabled.checked = True
    widget.frames.value = 10

    baseline = nvidia_smi("memory.used")
    peak = {"mib": int(baseline) if baseline else 0}

    def sample() -> None:
        used = nvidia_smi("memory.used")
        if used:
            peak["mib"] = max(peak["mib"], int(used))

    timer = qt.QTimer()
    timer.timeout.connect(sample)
    timer.start(2000)
    start = time.time()
    try:
        success = run_helper(widget.helper, widget._run)
    finally:
        timer.stop()
    (validation.output / "respiration-log.txt").write_text(
        widget.helper.log.toPlainText(), encoding="utf-8"
    )
    if not success:
        raise RuntimeError("simulation failed; see respiration-log.txt")
    vtp = sorted((folder / "VTP").glob("*.vtp"))
    ct = sorted((folder / "CT").glob("*_ct.mha"))
    usd = sorted(folder.glob("*.usd*"))
    sequences = [
        n.GetName() for n in slicer.util.getNodesByClass("vtkMRMLSequenceNode")
    ]
    if len(ct) != 10 or not usd or len(sequences) < 2:
        raise RuntimeError(
            f"{len(vtp)} VTP, {len(ct)} CT, {len(usd)} USD, sequences {sequences}"
        )
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd[0]))
    return {
        "minutes": round((time.time() - start) / 60, 1),
        "vtp": len(vtp), "ct_stages": len(ct), "usd": str(usd[0]),
        "usd_time_codes": [stage.GetStartTimeCode(), stage.GetEndTimeCode()],
        "sequences": sequences,
        "gpu_memory_baseline_mib": int(baseline) if baseline else None,
        "gpu_memory_peak_mib": peak["mib"],
    }


# ------------------------------------------------------------------ cpu


def phase_cpu(validation: Validation) -> None:
    validation.info["environment"] = environment_info()

    def no_gpu() -> str:
        import torch

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise RuntimeError("start Slicer with CUDA_VISIBLE_DEVICES=-1")
        if torch.cuda.is_available():
            raise RuntimeError("CUDA is still visible")
        return "CUDA hidden"

    validation.check("CUDA hidden", no_gpu)

    def surface() -> dict[str, Any]:
        success, target, widget = export_usd(validation, "surface")
        if not success:
            raise RuntimeError("export failed without a GPU")
        count = len(widget.helper.logic.visible_models())
        return check_usd(target, count, "Mesh")

    validation.check("USD export without GPU: surface", surface)

    def hexahedra() -> dict[str, Any]:
        model = first_lung_model()
        success, target, _ = export_usd(validation, "hexahedra", single=model)
        if not success:
            raise RuntimeError("hexahedra export failed without a GPU")
        return {**check_usd(target, 1, "Mesh"), "model": model}

    validation.check("USD export without GPU: one model, hexahedra 4 mm", hexahedra)

    def tetrahedra() -> str:
        success, _, widget = export_usd(
            validation, "tetrahedra", single=first_lung_model()
        )
        if success:
            raise RuntimeError("tetrahedra export unexpectedly succeeded without CUDA")
        lines = [
            line for line in widget.helper.log.toPlainText().splitlines()
            if "Error" in line or "error" in line or "CUDA" in line or "cuda" in line
        ]
        return "failed as expected; last error lines: " + " | ".join(lines[-3:])

    validation.check("USD export without GPU: tetrahedra reports failure", tetrahedra)


def main() -> None:
    validation = Validation(OUTPUT, PHASE)
    patch_dialogs(validation)
    validation.log(f"phase {PHASE}, output {validation.output}")
    phases = {"setup": phase_setup, "test": phase_test, "cpu": phase_cpu}
    try:
        phases[PHASE](validation)
    except Exception:
        validation.log(traceback.format_exc())
        validation.checks.append({"name": "phase", "passed": False})
    code = validation.finish()
    sys.stdout.flush()
    slicer.util.exit(code)


main()
