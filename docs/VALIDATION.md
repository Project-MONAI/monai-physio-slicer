# Validation status

Release **2026.11.1**, validated on 2026-11-05.

## Environment

| | |
| --- | --- |
| OS | Windows 11 Enterprise (10.0.26200), x64 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB), driver 596.36 |
| Slicer | 5.10.0 (r34045), 5.12.4 (r34645), 5.13.0 Preview of 2026-11-03; each a fresh installation |
| Python | 3.12 (bundled with each Slicer) |
| Installed by setup | monai-physio 2026.11.1, torch 2.14.1+cu132, torchvision 0.29.1+cu132, torch-scatter 2.1.2+pt214cu132 (this repository's `Data` release), torch-geometric 2.8.1, nvidia-physicsnemo 2.2.2, warp-lang 1.19.0, usd-core 26.11, pyvista 0.49.2, monai 1.6.1 |
| Kept from Slicer | numpy 2.4.6, vtk 9.6.2 |

## Results

All checks passed in all three Slicer releases.

| Check | 5.10.0 | 5.12.4 | 5.13.0 |
| --- | --- | --- | --- |
| **Set up MONAI-Physio in Slicer...** on a fresh Slicer, PyTorch 2.14.1+cu132 | pass (6.4 min) | pass (4.8 min) | pass (6.0 min) |
| Both modules register under **Simulation**; sample data registered; MONAI-Physio version accepted | pass | pass | pass |
| MONAIPhysioExportUSD self-test | pass | pass | pass |
| MONAIPhysioRespiratorySimulation self-test | pass | pass | pass |
| USD export, all 88 sample models, surface: 88 meshes, meters, Y-up | pass (26 s) | pass (43 s) | pass (27 s) |
| USD export, one lung lobe, tetrahedra 4 mm: TetMesh, GPU | pass (18 s) | pass (22 s) | pass (59 s) |
| Respiratory simulation of the sample CT, 10 stages, animated USD and 4D-CT: 12 VTP surfaces, 10 CT volumes, USD time codes 0-9, both sequences loaded | pass (9.0 min) | pass (10.0 min, first run incl. downloads) | pass (outputs in 9 min; 26 min until the worker exited, see below) |
| GPU memory during respiratory simulation, above idle | 14.6 GiB | 14.8 GiB | 14.8 GiB |
| Without a GPU (`CUDA_VISIBLE_DEVICES=-1`): surface export of 88 models | pass | pass | pass |
| Without a GPU: hexahedra export, one lung lobe, 4 mm | pass | pass | pass |
| Without a GPU: tetrahedra export fails and reports the error in the log | pass | pass | pass |
| Unit tests (`python -m pytest tests`), run once outside Slicer | 41 passed | | |

The first respiratory simulation on this machine started with empty weight
caches and downloaded the lung-motion and registration networks and
NV-Segment-CTMR; later runs reused them. Times were measured with other
validation runs sharing the machine, so treat them as indicative.

Peak GPU memory was sampled with `nvidia-smi` every 2 s during the
simulation and is reported above the idle baseline. Plan for a CUDA GPU with
at least 16 GB for respiratory simulation; smaller GPUs are untested.

Without a GPU, tetrahedra export stops with `RuntimeError: No CUDA GPUs are
available` from PyTorch in the module log; the module does not yet report a
shorter explanation.

Known issue on 5.13.0 Preview: the respiratory-simulation worker wrote all of
its outputs after 9 minutes, as on 5.10 and 5.12, but its process then took
about 17 more minutes to exit (in `os._exit`, idle, still holding GPU
memory), so the module showed the run as busy until then. Seen once; not seen
on 5.10.0 or 5.12.4.

Slicer's error log during the runs held only warnings from loading the
predicted VTP surfaces (`vtkMRMLModelStorageNode::ReadDataInternal`) and a
subject-hierarchy message when the validation script cleared the scene; no
error dialogs were shown.

## How to rerun

[`tools/slicer_validation.py`](../tools/slicer_validation.py) drives both
modules through their widgets in an installed Slicer. Setup installs several
GB into Slicer's Python, so use a Slicer installation you can restore (for
example, copy the installation folder first and restore it afterwards), and
run with `--disable-settings` so your Slicer settings are not changed. For
each Slicer release, run three sessions in order, setting the phase each time:

```bat
set MONAI_PHYSIO_VALIDATION_OUTPUT=C:\tmp\slicer-validation\5.12.4
set MONAI_PHYSIO_VALIDATION_PHASE=setup
"%LOCALAPPDATA%\slicer.org\3D Slicer 5.12.4\Slicer.exe" --no-splash --disable-settings ^
  --additional-module-paths MONAIPhysioExportUSD MONAIPhysioRespiratorySimulation ^
  --python-script tools\slicer_validation.py
rem then MONAI_PHYSIO_VALIDATION_PHASE=test, and finally
rem MONAI_PHYSIO_VALIDATION_PHASE=cpu with CUDA_VISIBLE_DEVICES=-1
```

Each session writes `<output>\<phase>\results.json`, logs and screenshots,
then exits with status 0 only if every check passed. Downloads go to
`~/.monai_physio`, the Hugging Face cache and Slicer's cache folder.

## Tested elsewhere

Slicer's extension factory builds the extension nightly on Windows, macOS and
Linux for each Slicer release that lists it, and runs the generic module tests
and both self-tests there ([CDash](https://slicer.cdash.org)). It has no GPU
and does not install MONAI-Physio.

## Not yet validated

- End-to-end workflows on macOS and Linux.
- GPUs other than the one above, including GPUs with less than 16 GB.
- Visual comparison of exported USD appearance in Omniverse applications.
- AutoRun of USD export in an interactive session (covered by the unit tests
  with mocked transport only).
- The package built by the extension factory, installed from the Extensions
  Manager.
