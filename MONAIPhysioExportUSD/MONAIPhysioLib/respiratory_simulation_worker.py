# SPDX-License-Identifier: Apache-2.0
"""Simulate respiratory lung motion for one chest CT, for Slicer.

Follows MONAI-Physio 2026.11's
``tutorials/tutorial_00_infer_respiratory_motion_tcia_lung_3d_ct.ipynb`` on a
CT the user selected in Slicer: segment the lung lobes with NV-Segment-CTMR, fit the PCA shape model, predict
a respiratory cycle (10 stages by default) with the pretrained MeshGraphNet, and optionally warp
the CT through each stage (4D-CT) and write an animated USD. Only the
pretrained network is downloaded (once, into the weights folder). Writes
``result.json`` in the VTP folder listing the outputs. Run with PythonSlicer::

    python respiratory_simulation_worker.py --image CT.nrrd --name NAME
        --weights-dir DIR --vtp-dir DIR [--ct-dir DIR]
        [--usd-dir DIR [--usd-colors slicer|anatomy] [--slicer-color R G B]]

Needs a CUDA GPU that the installed PyTorch supports, for NV-Segment-CTMR;
without one the run stops with a message before downloading anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, cast

# monai-physio's import-time install hints (CuPy, uv extras) do not apply to
# Slicer's pip-managed Python; keep them out of the module log.
for _hint in (r"CuPy is not installed", r"monai-physio was installed with pip"):
    warnings.filterwarnings("ignore", message=_hint, category=UserWarning)

DEFAULT_FRAMES = 10
DEFAULT_CYCLES_PER_MINUTE = 12.0
# MONAI-Physio names stage files s{int(stage * 100):03d}; stages closer than
# 0.02 can truncate to the same name, so keep at most 50 frames per cycle.
MAX_FRAMES = 50


def cycle_stages(frames: int) -> list[float]:
    """Evenly spaced normalized stages over one respiratory cycle."""
    if not 2 <= frames <= MAX_FRAMES:
        raise ValueError(f"Frames per cycle must be 2 to {MAX_FRAMES}.")
    return [round(i / frames, 6) for i in range(frames)]


def frames_per_second(frames: int, cycles_per_minute: float) -> float:
    """Playback rate showing one cycle every 60 / cycles_per_minute seconds."""
    if cycles_per_minute <= 0:
        raise ValueError("Cycles per minute must be positive.")
    return frames * cycles_per_minute / 60.0


STAGES = cycle_stages(DEFAULT_FRAMES)
USD_COLORS = ("slicer", "anatomy")
# Slicer's GenericAnatomyColors lung color, used when none is given.
SLICER_LUNG_COLOR = (197 / 255, 165 / 255, 145 / 255)

# Values from tutorials/parameters_tcia_lung_4d_ct.py. The lung shape model and
# the motion network were built from NV-Segment-CTMR's five lung lobes, so new
# CTs are segmented the same way and only the lobes are kept.
ANATOMY_GROUP = "lung"
SURFACE_REDUCTION_RATE = 0.0
ICP_TRANSFORM_TYPE = "Affine"
MASK_DILATION_MM = 20.0
DISTANCEMAP_SQUARED_MAX = (1.25 * 20.0) ** 2
NUMBER_OF_PCA_COMPONENTS = 6
SMOOTHING_SIGMA_MM = 10.0
# Margin kept around the lungs when cropping the CT for the shape-model fit,
# beyond the 25 mm at which its distance maps saturate, and the isotropic
# pitch the fit registers on (see fit_grid_image).
FIT_MARGIN_MM = 40.0
FIT_SPACING_MM = 1.5

# Points in pca_mean_surface.vtp of the lung model MONAI-Physio downloads now.
# DownloadData skips the download once mgn_stage_model.pt exists, so an older
# model already in the weights folder is never replaced.
LUNG_MODEL_POINTS = 194_010


def progress(percent: float, message: str) -> None:
    """Report progress to the Slicer module, which drives its progress bar."""
    print(f"PROGRESS {int(round(percent))} {message}", flush=True)


class PhaseProgress:
    """Map MONAI-Physio's progress callback onto a share of the progress bar.

    MONAIPhysioBase.set_progress_callback reports each workflow run as a
    fraction from 0 to 1; phase() sets the percent range the next run fills.
    Only whole-percent changes are printed.
    """

    def __init__(self) -> None:
        self.start, self.end = 0.0, 100.0
        self.last: Optional[int] = None

    def phase(self, start: float, end: float) -> None:
        self.start, self.end, self.last = start, end, None

    def __call__(self, fraction: float, message: str) -> None:
        percent = int(self.start + (self.end - self.start) * fraction)
        if percent != self.last:
            self.last = percent
            progress(percent, message)


def crop_to_mask(images: list[Any], mask_array: Any, margin_mm: float) -> list[Any]:
    """Crop images sharing one grid to the mask's bounding box plus a margin.

    The shape-model fit resamples its image to isotropic voxels, pads it and
    registers distance maps on that grid: for a whole chest CT, hundreds of
    millions of voxels and tens of GB of memory, while the organ and its
    surroundings fit in a fraction of it. The distance maps saturate within
    a few centimeters of the organ surface, so a wider margin adds nothing to
    the fit. Physical coordinates are kept, so the fitted mesh is unaffected.
    """
    import itk
    import numpy as np

    spacing = np.array(images[0].GetSpacing())  # x, y, z
    size = np.array(images[0].GetLargestPossibleRegion().GetSize())
    nonzero = np.argwhere(mask_array)  # z, y, x
    margin = np.ceil(margin_mm / spacing).astype(int)
    lower = np.maximum(nonzero.min(axis=0)[::-1] - margin, 0)
    upper = np.minimum(nonzero.max(axis=0)[::-1] + margin + 1, size)
    region = itk.ImageRegion[3]()
    region.SetIndex([int(v) for v in lower])
    region.SetSize([int(v) for v in upper - lower])
    return [
        itk.region_of_interest_image_filter(image, region_of_interest=region)
        for image in images
    ]


def fit_grid_image(image: Any, mask_array: Any, margin_mm: float, spacing_mm: float) -> Any:
    """The CT cropped around the organ and resampled to isotropic spacing_mm.

    WorkflowFitStatisticalModelToPatient registers on its image's grid,
    resampling a non-isotropic image to the CT's finest pitch: for a chest CT
    (0.7-0.8 mm) that is 100-250 million voxels once padded, and its
    distance-map registration then needs tens of GB of memory. An isotropic
    image is used as given, so a coarser grid here bounds that memory; the
    fit's distance maps vary smoothly and Greedy downsamples them anyway.
    """
    from monai_physio import ProcessImages

    (cropped,) = crop_to_mask([image], mask_array, margin_mm)
    return ProcessImages().make_isotropic_image(cropped, spacing_mm)


def read_ct(image_file: Path, quiet: bool = False) -> Any:
    """Read the CT as signed 16-bit voxels, converting other voxel types.

    MONAI-Physio's ITK filters are wrapped for a few voxel types only: a CT
    stored as 32-bit int (Slicer's CTChest sample), for example, fails in the
    first resampling of the segmenter. Values are rounded and clipped to the
    signed short range, which holds every Hounsfield unit. quiet skips the
    message, for a second read of a CT already reported.
    """
    import itk
    import numpy as np

    image = itk.imread(str(image_file))
    array = itk.array_view_from_image(image)
    if array.dtype == np.int16:
        return image
    limits = np.iinfo(np.int16)
    clipped = int(np.count_nonzero((array < limits.min) | (array > limits.max)))
    converted = itk.image_from_array(
        np.clip(np.rint(array), limits.min, limits.max).astype(np.int16)
    )
    converted.CopyInformation(image)
    if quiet:
        return converted
    print(
        f"Converted the CT voxels from {array.dtype} to signed short (int16)"
        + (f"; {clipped} voxels outside -32768..32767 were clipped." if clipped else "."),
        flush=True,
    )
    return converted


def fit_in_child_process(script: Path, argv: list[str]) -> None:
    """Run the segmentation and shape-model fit of script in a child process.

    MONAI-Physio's segmentation and registration leave about 15 GB of memory
    allocated that neither garbage collection nor emptying PyTorch's cache
    returns; only the end of the process does. Fitting in a child, which
    writes its results to files and exits, lets the prediction and CT warp
    start from a clean process. The child's output goes to this process's
    output, so its progress reaches the Slicer module unchanged.
    """
    import subprocess

    sys.stdout.flush()
    code = subprocess.call(
        [sys.executable, "-u", str(script), *argv, "--phase", "fit"],
        stdin=subprocess.DEVNULL,
    )
    if code != 0:
        raise RuntimeError(
            f"Segmentation and shape-model fit failed (exit code {code}); "
            "see the messages above."
        )


def exit_with_parent() -> None:
    """End this process when the process that started it ends.

    Cancel in the Slicer module kills the worker it started, which does not
    end that worker's own child process: the child watches its parent instead.
    """
    import threading
    import time

    import psutil

    parent = psutil.Process(os.getppid())

    def watch() -> None:
        while True:
            try:
                if not parent.is_running():
                    os._exit(1)
            except psutil.Error:
                os._exit(1)
            time.sleep(1.0)

    threading.Thread(target=watch, daemon=True).start()


def release_memory() -> None:
    """Return freed Python objects and PyTorch's cached GPU blocks.

    After segmentation the segmenter's network and buffers are no longer
    needed, but PyTorch keeps its caching allocator's blocks (about 9 GB of
    process memory for NV-Segment-CTMR on a chest CT) until asked; the fit
    that follows needs that memory.
    """
    import gc

    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_stages(
    infer_movement: Any,
    stages: list[float],
    phases: PhaseProgress,
    start: float,
    end: float,
    **arguments: Any,
) -> dict[str, list[Any]]:
    """Run WorkflowInferMovement.process_time_series one stage at a time.

    process_time_series keeps every stage's full-resolution displacement
    field (about 1 GB per stage for a 512x512 chest CT) in its result until
    it returns, which a long series cannot afford. One stage per call frees
    each field before the next is built; the outputs are the same.
    """
    import gc

    combined: dict[str, list[Any]] = {
        "predicted_surfaces": [], "warped_images": [], "stage_meshes": [],
    }
    share = (end - start) / len(stages)
    for index, stage in enumerate(stages):
        phases.phase(start + index * share, start + (index + 1) * share)
        result = infer_movement.process_time_series(stages=[stage], **arguments)
        for key in combined:
            combined[key].extend(result[key])
        del result
        gc.collect()
    return combined


def stale_lung_model_warning(
    model_dir: Path, mean_points: int, pca_model: dict[str, Any]
) -> Optional[str]:
    """Warning text when the cached lung model is not the current one, else None.

    The fit with an older model pins the shape coefficients at their bounds and
    the predicted motion is small and does not cycle.
    """
    widths = {len(component) for component in pca_model.get("components", [])}
    if mean_points == LUNG_MODEL_POINTS and widths == {3 * mean_points}:
        return None
    return (
        f"Warning: the lung model in {model_dir} is out of date ({mean_points} "
        f"surface points; the current model has {LUNG_MODEL_POINTS}). It gives "
        "little or no breathing motion. Delete that folder so the next run "
        "downloads the current model."
    )


def lung_segmenter(log_level: int) -> Any:
    """Return NV-Segment-CTMR, the segmenter the lung models were built from.

    It runs only on a CUDA GPU. Its weights are downloaded once by
    huggingface_hub and carry a non-commercial research license, which the
    segmenter logs on first use.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "NV-Segment-CTMR needs a CUDA GPU that the installed PyTorch "
            "supports. Pick a PyTorch build for this GPU in the setup menu."
        )
    from monai_physio import SegmentNVSegmentCTMRI

    print("Segmenting the lung lobes with NV-Segment-CTMR.", flush=True)
    return SegmentNVSegmentCTMRI(log_level=log_level)


def lobe_labels(taxonomy: Any) -> dict[int, str]:
    """The lung lobes among the segmenter's lung labels.

    The lung group also holds the airway, tumors and whole-lung labels, which
    the shape model and the motion network do not include.
    """
    return {
        label_id: label_name
        for label_id, label_name in taxonomy.labels_in_group(ANATOMY_GROUP).items()
        if "_lobe_" in label_name
    }


def run(
    image_file: Path,
    name: str,
    weights_dir: Path,
    vtp_dir: Path,
    ct_dir: Optional[Path] = None,
    usd_dir: Optional[Path] = None,
    usd_colors: str = "slicer",
    slicer_color: tuple[float, float, float] = SLICER_LUNG_COLOR,
    frames: int = DEFAULT_FRAMES,
    cycles_per_minute: float = DEFAULT_CYCLES_PER_MINUTE,
    usd_name: Optional[str] = None,
    phase: str = "all",
) -> dict[str, Any]:
    """Run the simulation and return the result listing written to result.json.

    phase "fit" segments the CT and fits the shape model, writing the shape
    parameters, the fitted mesh and the organ mask to the VTP folder, and
    stops; "predict" continues from those files; "all" does both. main runs
    the fit in a child process: the segmentation and registration leave
    memory allocated that only the end of the process returns.
    """
    import itk
    import numpy as np
    import pyvista as pv

    from monai_physio import (
        DownloadData,
        MONAIPhysioBase,
        ProcessSurfaceMesh,
        WorkflowConvertImageToVTK,
        WorkflowConvertVTKToUSD,
        WorkflowFitStatisticalModelToPatient,
        WorkflowInferMovement,
        WorkflowInferPhysicsNeMo,
    )

    stages = cycle_stages(frames)
    fps = frames_per_second(frames, cycles_per_minute)
    log_level = logging.INFO
    logging.basicConfig(level=log_level, stream=sys.stdout)
    for directory in (vtp_dir, ct_dir, usd_dir):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
    model_dir = weights_dir / "physicsnemo_mgn_lung_motion"
    # process_time_series names predicted surfaces after this file's stem.
    pca_coefficients_file = vtp_dir / f"{name}_surface.json"
    fitted_reference_mesh_file = vtp_dir / f"{name}_ssm_surface.vtp"
    mask_file = vtp_dir / f"{name}_lung_mask.mha"
    phases = PhaseProgress()
    MONAIPhysioBase.set_progress_callback(phases)
    if phase != "predict":
        # Fails without a usable GPU, so check before downloading anything.
        segmentation_method = lung_segmenter(log_level)

        progress(1, "Loading the CT and pretrained network")
        if not (model_dir / "mgn_stage_model.pt").exists():
            progress(2, f"Downloading the pretrained network into {weights_dir}")
            DownloadData.DownloadPhysicsNeMoMGNLungMotionData(weights_dir)
        pca_model_file = model_dir / "pca_model.json"
        pca_mean_file = model_dir / "pca_mean_surface.vtp"
        for required in (model_dir / "mgn_stage_model.pt", pca_model_file, pca_mean_file):
            if not required.exists():
                raise FileNotFoundError(f"Missing pretrained network file: {required}")
        pca_mean_surface = cast(pv.DataSet, pv.read(str(pca_mean_file)))
        with pca_model_file.open(encoding="utf-8") as f:
            pca_model = json.load(f)
        stale = stale_lung_model_warning(model_dir, pca_mean_surface.n_points, pca_model)
        if stale:
            print(stale, flush=True)
        patient_image = read_ct(image_file)

        progress(5, "Segmenting the lung lobes")
        phases.phase(5, 20)
        lung_surface_file = vtp_dir / f"{name}_lung_surface.vtp"
        segmentation_result = WorkflowConvertImageToVTK(
            segmentation_method=segmentation_method, log_level=log_level
        ).process(
            input_image=patient_image,
            anatomy_groups=[ANATOMY_GROUP],
            surface_reduction_rate=SURFACE_REDUCTION_RATE,
            extract_label_surfaces=True,
            extract_group_surfaces=False,
        )
        lobes = lobe_labels(segmentation_method.taxonomy)
        del segmentation_method
        release_memory()
        lobe_surfaces = {
            label_name: surface
            for label_name, surface in segmentation_result["label_surfaces"].items()
            if label_name in lobes.values()
        }
        if not lobe_surfaces:
            raise RuntimeError("NV-Segment-CTMR found no lung lobes in this CT.")
        print(f"Lung lobes: {', '.join(sorted(lobe_surfaces))}", flush=True)
        lung_surface = ProcessSurfaceMesh.combine_surfaces(lobe_surfaces)
        lung_surface.save(str(lung_surface_file))
        lung_labelmap = segmentation_result["labelmap"]
        # Binary lobe mask, for the fit's distance map and the warp's exterior
        # mask. The whole-body labelmap is 0 inside every labelled structure in the
        # fit's distance map, letting the lung model grow into its neighbours.
        lung_label_ids = list(lobes)
        lung_only_mask = itk.image_from_array(
            np.isin(itk.array_from_image(lung_labelmap), lung_label_ids).astype(np.uint8)
        )
        lung_only_mask.CopyInformation(lung_labelmap)

        progress(20, "Fitting the lung shape model")
        phases.phase(20, 60)
        fit_image = fit_grid_image(
            patient_image, itk.array_view_from_image(lung_only_mask) > 0,
            FIT_MARGIN_MM, FIT_SPACING_MM,
        )
        fit_workflow = WorkflowFitStatisticalModelToPatient(
            template_model=pca_mean_surface,
            patient_models=[lung_surface],
            patient_image=fit_image,
            patient_labelmap=lung_only_mask,
            log_level=log_level,
        )
        fit_workflow.set_use_pca_registration(
            use_pca_registration=True,
            pca_model=pca_model,
            number_of_pca_components=NUMBER_OF_PCA_COMPONENTS,
            use_surface=False,
        )
        fit_workflow.set_icp_transform_type(ICP_TRANSFORM_TYPE)
        fit_workflow.set_mask_dilation_mm(MASK_DILATION_MM)
        fit_workflow.set_distancemap_squared_max(DISTANCEMAP_SQUARED_MAX)
        fit_result = fit_workflow.process()
        pca_coefficients = fit_workflow.pca_coefficients
        assert pca_coefficients is not None
        with pca_coefficients_file.open(mode="w", encoding="utf-8") as f:
            json.dump(pca_coefficients.tolist(), f)
        fit_result["fitted_reference_mesh"].save(str(fitted_reference_mesh_file))
        itk.imwrite(lung_only_mask, str(mask_file), compression=True)
        if phase == "fit":
            MONAIPhysioBase.set_progress_callback(None)
            progress(60, "Shape model fitted")
            return {}
        del fit_workflow, fit_result, fit_image, segmentation_result, lung_labelmap
        release_memory()
    else:
        for required in (pca_coefficients_file, fitted_reference_mesh_file, mask_file):
            if not required.exists():
                raise FileNotFoundError(f"Missing shape-model fit output: {required}")
        # Only the CT warp needs the CT and the organ mask.
        patient_image = read_ct(image_file, quiet=True) if ct_dir else None
        lung_only_mask = itk.imread(str(mask_file)) if ct_dir else None

    predict_end = 92 if usd_dir is not None else 98
    progress(
        60, f"Predicting {frames} respiratory stages"
        + (" and warping the CT" if ct_dir else "")
    )
    infer_workflow = WorkflowInferPhysicsNeMo(
        model_directory=model_dir, epoch=None, log_level=log_level
    )
    infer_result = predict_stages(
        WorkflowInferMovement(infer_workflow, log_level=log_level),
        stages, phases, 60, predict_end,
        shape_parameters=pca_coefficients_file,
        output_directory=vtp_dir,
        fitted_reference_mesh=fitted_reference_mesh_file,
        reference_image=patient_image if ct_dir else None,
        warp_interpolation="linear",
        warp_background_value=-1000.0,
        smoothing_sigma_mm=SMOOTHING_SIGMA_MM,
        exterior_mask=lung_only_mask if ct_dir else None,
        exterior_falloff_distance_mm=10.0,
    )
    images = []
    if ct_dir is not None:
        for stage, warped_file in zip(stages, infer_result["warped_images"]):
            moved = ct_dir / f"{name}_s{int(stage * 100):03d}_ct.mha"
            Path(warped_file).replace(moved)
            images.append(str(moved))

    usd_file = None
    if usd_dir is not None:
        progress(predict_end, f"Writing the USD with {usd_colors} colors")
        phases.phase(predict_end, 99)
        usd_workflow = WorkflowConvertVTKToUSD(
            input_meshes=infer_result["stage_meshes"],
            # A file name; its stem also names the root prim /World/<stem>.
            usd_project_name=usd_name or f"{name}_respiration.usd",
            output_directory=usd_dir,
            appearance="anatomy" if usd_colors == "anatomy" else "solid",
            anatomy_type="lung" if usd_colors == "anatomy" else None,
            solid_color=tuple(slicer_color),
            separate_by_connectivity=True,
            frames_per_second=fps,
            log_level=log_level,
        )
        usd_file = str(usd_workflow.process()["usd_file"])

    MONAIPhysioBase.set_progress_callback(None)
    result = {
        "stages": stages,
        "frames_per_second": fps,
        "surfaces": [str(p) for p in infer_result["predicted_surfaces"]],
        "images": images,
        "usd": usd_file,
    }
    (vtp_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    progress(100, "Done")
    return result


_GPU_PROBE = """
import torch
if not torch.cuda.is_available():
    print("cpu (no CUDA device)")
else:
    major, minor = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    sm = major * 10 + minor
    # Native code for this GPU, or PTX from an older arch the driver can JIT.
    native = f"sm_{sm}" in arches
    ptx = any(a.startswith("compute_") and int(a[8:]) <= sm for a in arches)
    name = torch.cuda.get_device_name(0)
    if native or ptx:
        print(f"cuda {name} (sm_{sm}, torch {torch.__version__})")
    else:
        print(f"unsupported {name} (sm_{sm}; torch {torch.__version__} "
              f"has {' '.join(arches)})")
"""


def gpu_status() -> str:
    """Whether the installed PyTorch can run on this machine's GPU.

    PyTorch reports CUDA as available even when its build has no kernels for
    the GPU (e.g. a cu126 build on an RTX 50 / Blackwell card), and then fails
    mid-run with "no kernel image is available". Probing in a separate
    process leaves CUDA uninitialized here, so it can still be hidden.
    """
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-c", _GPU_PROBE], capture_output=True, text=True,
            timeout=300,
        )
        lines = result.stdout.strip().splitlines()
        return lines[-1] if lines else f"cpu (probe failed: {result.stderr[-200:]})"
    except Exception as exc:
        return f"cpu (probe failed: {exc})"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--name", default="CT")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--vtp-dir", type=Path, required=True)
    parser.add_argument("--ct-dir", type=Path)
    parser.add_argument("--usd-dir", type=Path)
    parser.add_argument("--usd-colors", choices=USD_COLORS, default="slicer")
    parser.add_argument("--usd-name",
                        help="USD file name (default <name>_respiration.usd)")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="frames per respiratory cycle")
    parser.add_argument("--cycles-per-minute", type=float,
                        default=DEFAULT_CYCLES_PER_MINUTE,
                        help="breathing rate, sets the playback frames per second")
    parser.add_argument(
        "--slicer-color", type=float, nargs=3, default=SLICER_LUNG_COLOR,
        metavar=("R", "G", "B"), help="USD solid color in [0, 1] (slicer colors)",
    )
    # Internal: the step a child process runs (see fit_in_child_process).
    parser.add_argument("--phase", choices=("fit", "predict", "all"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.phase == "fit":
        # The parent probed the GPU; CUDA_VISIBLE_DEVICES is inherited.
        exit_with_parent()
    else:
        gpu = gpu_status()
        print(f"PyTorch device: {gpu}", flush=True)
        if gpu.startswith("unsupported"):
            # Must be set before torch initializes CUDA in this process.
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print(
                "The installed PyTorch has no kernels for this GPU. Pick a "
                "matching PyTorch build in the setup menu.",
                flush=True,
            )
        result_file = args.vtp_dir / "result.json"
        if result_file.exists():
            result_file.unlink()
    phase = args.phase
    if phase is None:
        fit_in_child_process(
            Path(__file__).resolve(), list(sys.argv[1:] if argv is None else argv)
        )
        phase = "predict"
    run(
        args.image, args.name, args.weights_dir, args.vtp_dir, args.ct_dir,
        args.usd_dir, args.usd_colors, tuple(args.slicer_color),
        args.frames, args.cycles_per_minute, args.usd_name, phase=phase,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # Skip interpreter teardown, where background threads from heavy
    # dependencies can keep the process (and the Slicer GUI) waiting.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
