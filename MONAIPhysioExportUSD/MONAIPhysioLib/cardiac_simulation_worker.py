# SPDX-License-Identifier: Apache-2.0
"""Simulate cardiac motion for one chest CT, for Slicer.

Follows MONAI-Physio 2026.11's
``tutorials/tutorial_00_infer_cardiac_motion_duke_heart_4d_labelmaps.ipynb``
(fit the Duke heart shape model, predict a cardiac cycle with the pretrained
MeshGraphNet) on a CT the user selected in Slicer, segmenting the heart the way
``tutorial_13_combine_heart_and_lung_motion_chest_3d_ct.py`` does, and
optionally warps the CT through each stage (4D-CT) and writes an animated USD.
Writes ``result.json`` in the VTP folder listing the outputs, in the same form
as the respiratory worker. Run with PythonSlicer::

    python cardiac_simulation_worker.py --image CT.nrrd --name NAME
        --weights-dir DIR --vtp-dir DIR [--ct-dir DIR]
        [--segmenter simpleware|nvsegment] [--simpleware-exe PATH]
        [--usd-dir DIR [--usd-colors slicer|anatomy] [--slicer-color R G B]]

The heart shape model and network were built from labelmaps made by the
ASCardio module of Synopsys Simpleware Medical, so ``simpleware`` (needs a
Simpleware Medical installation) gives the geometry the network was trained
on. ``nvsegment`` uses NV-Segment-CTMR (needs a CUDA GPU) and maps its heart
labels onto the Simpleware ids; it is an approximation, and on non-contrast
CTs NV-Segment-CTMR can return the heart as one oversized region.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, cast

# Shared with the respiratory worker, which sits in the same folder (on
# sys.path when this file is run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from respiratory_simulation_worker import (  # noqa: E402
    MAX_FRAMES,
    PhaseProgress,
    cycle_stages,
    exit_with_parent,
    fit_grid_image,
    fit_in_child_process,
    frames_per_second,
    gpu_status,
    predict_stages,
    progress,
    read_ct,
    release_memory,
)

DEFAULT_FRAMES = 10
DEFAULT_CYCLES_PER_MINUTE = 60.0
SEGMENTERS = ("simpleware", "nvsegment")
USD_COLORS = ("slicer", "anatomy")
# Slicer's GenericAnatomyColors heart color, used when none is given.
SLICER_HEART_COLOR = (206 / 255, 110 / 255, 84 / 255)
WEIGHTS_SUBFOLDER = "physicsnemo_mgn_duke_heart_motion"

# Values from tutorials/parameters_duke_heart_4d_labelmaps.py (2026.11.1).
ICP_TRANSFORM_TYPE = "Similarity"
MASK_DILATION_MM = 10.0
DISTANCEMAP_SQUARED_MAX = (1.25 * 10.0) ** 2
NUMBER_OF_PCA_COMPONENTS = 10
SURFACE_SPACING_MM = 0.5
SURFACE_SMOOTHING_ITERATIONS = 20
# Simpleware ASCardio ids: 1-4 chambers, 5 myocardium, 6 heart wall, 7-10
# great vessels and coronaries. The shape model is the heart minus the
# chamber cavities and vessels, i.e. labels 5 and 6.
INTERIOR_OBJECT_IDS = [1, 2, 3, 4, 7, 8, 9, 10]
WHOLE_HEART_NAME = "heart_minus_interior_chambers"
# Margin kept around the heart when cropping the CT for the shape-model fit
# (several times the mask dilation, so the fit's distance maps see the heart's
# surroundings), and the isotropic pitch the fit registers on: the Duke
# tutorials register heart frames on a 1 mm grid.
FIT_MARGIN_MM = 40.0
FIT_SPACING_MM = 1.0
# Values from tutorial 13: the heart sits in tissue that barely moves with it,
# so its push and pull reaches only a quarter of the smoothing sigma outside it.
SMOOTHING_SIGMA_MM = 10.0
EXTERIOR_FALLOFF_MM = 0.25 * SMOOTHING_SIGMA_MM
EXTERIOR_DIRECTION_OFFSET_MM = 3.0
EXTERIOR_DIRECTION_TRANSITION_MM = 5.0

# NV-Segment-CTMR heart ids -> the Simpleware ids the model was built from.
# The left atrial appendage is part of the left atrial cavity; the whole-heart
# and heart-tissue labels form the heart wall left once the chambers are cut.
NVSEGMENT_TO_SIMPLEWARE = {
    151: 1,  # ventricle_left
    152: 2,  # ventricle_right
    149: 3,  # atrium_left
    108: 3,  # atrial_appendage_left
    153: 4,  # atrium_right
    154: 5,  # ventricle_myocardium_left
    115: 6,  # heart
    169: 6,  # heart_tissue
}


def remap_nvsegment_heart(labelmap: Any) -> Any:
    """NV-Segment-CTMR labelmap -> Simpleware-id heart labelmap (others 0)."""
    import itk
    import numpy as np

    source = itk.array_from_image(labelmap)
    heart = np.zeros(source.shape, dtype=np.uint8)
    for nv_id, simpleware_id in NVSEGMENT_TO_SIMPLEWARE.items():
        heart[source == nv_id] = simpleware_id
    result = itk.image_from_array(heart)
    result.CopyInformation(labelmap)
    return result


def heart_segmenter(name: str, simpleware_exe: Optional[str], log_level: int) -> Any:
    """Return the segmenter that produces the heart labelmap of the CT."""
    if name == "simpleware":
        from monai_physio import SegmentHeartSimplewareTrimmedBranches

        segmenter = SegmentHeartSimplewareTrimmedBranches(log_level=log_level)
        if simpleware_exe:
            segmenter.set_simpleware_executable_path(simpleware_exe)
        if not Path(segmenter.simpleware_exe_path).is_file():
            raise FileNotFoundError(
                "Simpleware Medical was not found at "
                f"{segmenter.simpleware_exe_path}. Set the path to "
                "ConsoleSimplewareMedical.exe, or choose NV-Segment-CTMR."
            )
        print("Segmenting the heart with Simpleware Medical ASCardio.", flush=True)
        return segmenter

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "NV-Segment-CTMR needs a CUDA GPU that the installed PyTorch "
            "supports. Pick a PyTorch build for this GPU in the setup menu."
        )
    from monai_physio import SegmentNVSegmentCTMRI

    print(
        "Segmenting the heart with NV-Segment-CTMR. The heart model was built "
        "from Simpleware ASCardio segmentations, so the fit is approximate.",
        flush=True,
    )
    return SegmentNVSegmentCTMRI(log_level=log_level)


def run(
    image_file: Path,
    name: str,
    weights_dir: Path,
    vtp_dir: Path,
    ct_dir: Optional[Path] = None,
    usd_dir: Optional[Path] = None,
    usd_colors: str = "slicer",
    slicer_color: tuple[float, float, float] = SLICER_HEART_COLOR,
    frames: int = DEFAULT_FRAMES,
    cycles_per_minute: float = DEFAULT_CYCLES_PER_MINUTE,
    usd_name: Optional[str] = None,
    segmenter_name: str = "simpleware",
    simpleware_exe: Optional[str] = None,
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
        SegmentHeartSimplewareTrimmedBranches,
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
    model_dir = weights_dir / WEIGHTS_SUBFOLDER
    # process_time_series names predicted surfaces after this file's stem.
    pca_coefficients_file = vtp_dir / f"{name}_heart.json"
    fitted_reference_mesh_file = vtp_dir / f"{name}_heart_ssm_surface.vtp"
    mask_file = vtp_dir / f"{name}_heart_mask.mha"
    phases = PhaseProgress()
    MONAIPhysioBase.set_progress_callback(phases)
    if phase != "predict":
        # Fails without Simpleware or a usable GPU, so check before downloading.
        segmenter = heart_segmenter(segmenter_name, simpleware_exe, log_level)

        progress(1, "Loading the CT and pretrained network")
        if not (model_dir / "mgn_stage_model.pt").exists():
            progress(2, f"Downloading the pretrained network into {weights_dir}")
            DownloadData.DownloadPhysicsNeMoMGNDukeHeartMotionData(weights_dir)
        pca_model_file = model_dir / "pca_model.json"
        pca_mean_file = model_dir / "pca_mean_surface.vtp"
        for required in (model_dir / "mgn_stage_model.pt", pca_model_file, pca_mean_file):
            if not required.exists():
                raise FileNotFoundError(f"Missing pretrained network file: {required}")
        pca_mean_surface = cast(pv.DataSet, pv.read(str(pca_mean_file)))
        with pca_model_file.open(encoding="utf-8") as f:
            pca_model = json.load(f)
        patient_image = read_ct(image_file)

        progress(5, "Segmenting the heart")
        segmentation = segmenter.segment(patient_image)["labelmap"]
        del segmenter
        release_memory()
        if segmenter_name == "nvsegment":
            heart_labelmap = remap_nvsegment_heart(segmentation)
        else:
            heart_labelmap = segmentation
        heart_array = itk.array_from_image(heart_labelmap)
        wall_ids = [
            int(v) for v in np.unique(heart_array)
            if v != 0 and int(v) not in INTERIOR_OBJECT_IDS
        ]
        if not wall_ids:
            raise RuntimeError("The segmenter found no heart wall in this CT.")
        found = sorted(int(v) for v in np.unique(heart_array) if v != 0)
        print(f"Heart labels (Simpleware ids): {found}", flush=True)
        if not {1, 2} <= set(found):
            print(
                "Warning: the ventricles were not segmented, so the fitted heart "
                "includes their cavities and the predicted motion is unreliable. "
                "The heart model was built from Simpleware ASCardio segmentations, "
                "which separate the chambers.",
                flush=True,
            )

        progress(15, "Extracting the heart surface")
        # The structure the shape model describes: the heart minus its chamber
        # cavities and great vessels. The Simpleware segmenter only supplies the
        # label names here; nothing is segmented again.
        surface_result = WorkflowConvertImageToVTK(
            SegmentHeartSimplewareTrimmedBranches(log_level=log_level),
            log_level=log_level,
        ).process(
            labelmap_image=heart_labelmap,
            groups={WHOLE_HEART_NAME: wall_ids},
            isotropic_spacing_mm=SURFACE_SPACING_MM,
            smoothing_iterations=SURFACE_SMOOTHING_ITERATIONS,
        )
        heart_surface = cast(pv.PolyData, surface_result["surfaces"][WHOLE_HEART_NAME])
        heart_surface.save(str(vtp_dir / f"{name}_heart_surface.vtp"))

        progress(20, "Fitting the heart shape model")
        phases.phase(20, 60)
        fit_image = fit_grid_image(
            patient_image, heart_array > 0, FIT_MARGIN_MM, FIT_SPACING_MM
        )
        fit_workflow = WorkflowFitStatisticalModelToPatient(
            template_model=pca_mean_surface,
            patient_models=[heart_surface],
            patient_image=fit_image,
            patient_labelmap=heart_labelmap,
            labelmap_interior_object_ids=INTERIOR_OBJECT_IDS,
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

        # Every heart label, chambers included, is the organ the warp's sliding
        # boundary is computed against (tutorial 13's cardiac interior mask).
        heart_mask = itk.image_from_array((heart_array > 0).astype(np.uint8))
        heart_mask.CopyInformation(heart_labelmap)
        itk.imwrite(heart_mask, str(mask_file), compression=True)
        if phase == "fit":
            MONAIPhysioBase.set_progress_callback(None)
            progress(60, "Shape model fitted")
            return {}
        del fit_workflow, fit_result, fit_image, segmentation
        release_memory()
    else:
        for required in (pca_coefficients_file, fitted_reference_mesh_file, mask_file):
            if not required.exists():
                raise FileNotFoundError(f"Missing shape-model fit output: {required}")
        # Only the CT warp needs the CT and the organ mask.
        patient_image = read_ct(image_file, quiet=True) if ct_dir else None
        heart_mask = itk.imread(str(mask_file)) if ct_dir else None

    predict_end = 92 if usd_dir is not None else 98
    progress(
        60, f"Predicting {frames} cardiac stages"
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
        exterior_mask=heart_mask if ct_dir else None,
        exterior_direction_offset_mm=EXTERIOR_DIRECTION_OFFSET_MM,
        exterior_direction_transition_mm=EXTERIOR_DIRECTION_TRANSITION_MM,
        exterior_falloff_distance_mm=EXTERIOR_FALLOFF_MM,
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
            usd_project_name=usd_name or f"{name}_heartbeat.usd",
            output_directory=usd_dir,
            appearance="anatomy" if usd_colors == "anatomy" else "solid",
            anatomy_type="heart" if usd_colors == "anatomy" else None,
            solid_color=tuple(slicer_color),
            # The shape model is one structure: keep each frame whole.
            separate_by_connectivity=False,
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
                        help="USD file name (default <name>_heartbeat.usd)")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help=f"frames per cardiac cycle (2 to {MAX_FRAMES})")
    parser.add_argument("--cycles-per-minute", type=float,
                        default=DEFAULT_CYCLES_PER_MINUTE,
                        help="heart rate, sets the playback frames per second")
    parser.add_argument("--segmenter", choices=SEGMENTERS, default="simpleware")
    parser.add_argument("--simpleware-exe",
                        help="path to ConsoleSimplewareMedical.exe")
    parser.add_argument(
        "--slicer-color", type=float, nargs=3, default=SLICER_HEART_COLOR,
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
        if args.segmenter == "nvsegment":
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
        args.frames, args.cycles_per_minute, args.usd_name,
        args.segmenter, args.simpleware_exe, phase=phase,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
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
