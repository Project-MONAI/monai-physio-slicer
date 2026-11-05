# Submit to Slicer's Extensions Catalog

Slicer's extension catalog is browsed in the application's **Extensions
Manager**. A catalog entry, one JSON file named after the extension's CMake
project (`MONAIPhysio.json`), is submitted to
[Slicer/ExtensionsIndex](https://github.com/Slicer/ExtensionsIndex). The Slicer
extension factory then builds, tests, packages and publishes the extension
every night for each Slicer release whose ExtensionsIndex branch lists it.
Current submissions use the JSON catalog entry
([schema v1.0.2](https://github.com/Slicer/ExtensionsIndex/blob/main/schemas/slicer-extension-catalog-entry-schema-v1.0.2.json)),
not the older `.s4ext` format.

The **MONAIPhysio** extension contains two Python scripted modules, both in
Slicer's **Simulation** category:

- **MONAI Physio USD Export** exports surface models, or tetrahedral or
  hexahedral volume meshes made from them, to an OpenUSD file for NVIDIA
  Omniverse.
- **MONAI-Physio Respiratory Simulation** predicts one respiratory cycle of
  lung motion from a chest CT and writes lung surfaces, a 4D-CT and an
  animated USD, loaded back as Slicer sequences.

## Repositories and branches

| Repository | Role |
| --- | --- |
| [Project-MONAI/monai-physio-slicer](https://github.com/Project-MONAI/monai-physio-slicer) | Upstream source. The catalog entries point here (`scm_url`). |
| [aylward/monai-physio-slicer](https://github.com/aylward/monai-physio-slicer) | Maintainer fork. Changes are staged here and merged upstream by pull request. |
| [Slicer/ExtensionsIndex](https://github.com/Slicer/ExtensionsIndex) | Slicer's catalog. Entries are submitted by pull request from the fork [aylward/Slicer-ExtensionsIndex](https://github.com/aylward/Slicer-ExtensionsIndex). |

The catalog entry is listed on three ExtensionsIndex branches, each built
against a different Slicer:

| ExtensionsIndex branch | Slicer | `scm_revision` (branch of Project-MONAI/monai-physio-slicer) |
| --- | --- | --- |
| `main` | Preview (nightly), currently 5.13 | `main` |
| `5.12` | 5.12 Stable | `5.12` |
| `5.10` | 5.10 Stable | `5.10` |

Stable entries use branch names, as Slicer's pull-request template suggests,
so an update needs only a push to that branch of this repository, not a new
ExtensionsIndex pull request. [`ExtensionsIndex/MONAIPhysio.json`](../ExtensionsIndex/MONAIPhysio.json)
holds the Preview entry; the Stable entries differ only in `scm_revision`.

### Repository name

Slicer's checklist asks for repository names of the form
`Slicer<ExtensionName>`. This repository is named `monai-physio-slicer`
instead because it lives in the Project-MONAI GitHub organization, which holds
many repositories: every MONAI-Physio repository there starts with
`monai-physio` so the related repositories are listed and found together.

## Versions and releases

Versions follow MONAI-Physio's calendar versions: release `2026.11.1` is the
first that requires MONAI-Physio 2026.11.1 (`MINIMUM_MONAI_PHYSIO` in
[runtime.py](../MONAIPhysioExportUSD/MONAIPhysioLib/runtime.py)). Slicer does
not use a version number for extensions; each package is named after the
commit it was built from.

To make a release:

1. In the fork, create the branch `release_<version>` (for example
   `release_2026.11.1`) and commit the changes, including the
   minimum MONAI-Physio version and [VALIDATION.md](VALIDATION.md).
2. Run the validation below on every Slicer release the catalog lists.
3. Push the branch to `aylward/monai-physio-slicer` and open a pull request to
   `Project-MONAI/monai-physio-slicer` `main`.
4. After merging, tag the merge commit with the bare version (`2026.11.1`, no
   leading `v`) and move the `5.10` and `5.12` branches to it. The Preview
   entry follows `main` without further action.

The extension factory picks up the new commits on its next nightly build.

## Before submitting or releasing

- Run the unit tests: `python -m unittest discover -s tests -v`.
- Run [`tools/slicer_validation.py`](../tools/slicer_validation.py) in each
  Slicer release the catalog lists. It drives both modules through their
  widgets: dependency setup, self-tests, USD export of the sample models,
  respiratory simulation of the sample CT, and a run with the GPU hidden. See
  [VALIDATION.md](VALIDATION.md) for the procedure and results. Use a Slicer
  installation you can restore, because setup installs several GB of
  packages.
- Record the tested Slicer versions, operating systems, GPUs and package
  versions in [VALIDATION.md](VALIDATION.md). Do not label untested versions
  supported.
- Check that the icon and screenshot URLs in [CMakeLists.txt](../CMakeLists.txt)
  resolve after the release is merged upstream.
- Keep the JSON and the top-level CMake metadata consistent: category
  `Simulation`, no build dependencies, repository URL.

### Testing by the extension factory

No separate CI workflow is used. The extension factory builds every catalog
extension nightly on Windows, macOS and Linux for each listed Slicer release,
runs `ctest`, and posts the results to CDash
([Preview](https://slicer.cdash.org/index.php?project=SlicerPreview),
[Stable](https://slicer.cdash.org/index.php?project=SlicerStable)). For this
extension, ctest runs Slicer's generic module tests (`WITH_GENERIC_TESTS`) and
each module's self-test (`slicer_add_python_unittest`), so module loading and
the self-tests are checked on all three platforms. The factory has no GPU and
does not install MONAI-Physio, so it does not run the end-to-end workflows;
`tools/slicer_validation.py` covers those. The `tests/` unit suite is run
locally.

## Requirements and downloads to disclose

Slicer's Tier 1 checklist requires that an extension downloads binaries only
from reliable sources and sends nothing without explicit consent. These
modules download only after the user asks:

| When | What | From |
| --- | --- | --- |
| **Set up MONAI-Physio in Slicer...**, after a confirmation | PyTorch and torchvision for the chosen build | download.pytorch.org |
| same | torch-scatter wheel | data.pyg.org, or this repository's [`Data` release](https://github.com/Project-MONAI/monai-physio-slicer/releases/tag/Data) (Windows, Python 3.12) |
| same | MONAI-Physio and its dependencies | PyPI |
| First respiratory simulation, after a confirmation | Pretrained lung-motion and registration networks (about 0.5 GB) into `~/.monai_physio/network_weights` | MONAI-Physio's download locations |
| First respiratory simulation | NV-Segment-CTMR weights (about 870 MB) into the Hugging Face cache | Hugging Face |
| **Download sample models** / **Download sample** | Sample scene (80 MB) / chest CT (89 MB) | this repository's `Data` release |

Setup downloads several GB, and the first respiratory simulation also
downloads the networks, so the first use is much slower than later ones.
Nothing is uploaded. The same information is in the README and in each
module's help text.

Hardware: surface and hexahedra export run on the CPU. Tetrahedra export and
respiratory simulation need a CUDA GPU supported by the installed PyTorch
build (Blackwell / RTX 50 GPUs need CUDA 12.8 or newer). Respiratory
simulation used about 15 GB of GPU memory in validation, so a GPU with 16 GB
or more is recommended.

## Submit the catalog entries

1. In your fork of Slicer/ExtensionsIndex, create one branch per target
   ExtensionsIndex branch (for example `add-monai-physio-main`,
   `add-monai-physio-5.12` and `add-monai-physio-5.10`), each starting from
   that upstream branch.
2. Copy [`ExtensionsIndex/MONAIPhysio.json`](../ExtensionsIndex/MONAIPhysio.json)
   to the root of each branch, and set `scm_revision` to `main`, `5.12` or
   `5.10` to match.
3. Check each file against the schema, commit it, push the branches to the
   fork, and open one pull request per branch to Slicer/ExtensionsIndex, using
   the **New extension** template. Describe both modules, the downloads and GPU
   requirements above, the repository name, and the validation performed.
   Check only the checklist items that are actually met.

After the pull requests are merged, follow the CDash build, test and package
results for each platform, then install from the **Extensions Manager** in
each Slicer release to confirm. Publication can take a few days after the
merge, and build or packaging failures must be fixed first.

## Repository settings for Tier 1

These are set on Project-MONAI/monai-physio-slicer by its administrators:

- the topic `3d-slicer-extension`;
- a homepage (the repository URL), with the README naming the extension,
  describing each module and the license, and showing a screenshot;
- Wiki and Projects disabled when unused. Releases stay visible because the
  `Data` release hosts the sample data and torch-scatter wheels.

## Official references

- [Extension development and distribution](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)
- [Extensions Index](https://github.com/Slicer/ExtensionsIndex)
- [Submission checklist](https://github.com/Slicer/ExtensionsIndex/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
- [Catalog entry schema](https://github.com/Slicer/ExtensionsIndex/blob/main/schemas/slicer-extension-catalog-entry-schema-v1.0.2.json)
