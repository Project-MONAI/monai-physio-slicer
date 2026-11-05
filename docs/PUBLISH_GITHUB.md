# GitHub repositories and contribution flow

| Remote | Repository | Use |
| --- | --- | --- |
| `upstream` | [Project-MONAI/monai-physio-slicer](https://github.com/Project-MONAI/monai-physio-slicer) | Canonical source; the Slicer catalog builds from it. |
| `aylward` | [aylward/monai-physio-slicer](https://github.com/aylward/monai-physio-slicer) | Maintainer fork; changes are pushed here. |

Default branch: `main`. License: Apache-2.0 ([LICENSE.txt](../LICENSE.txt)).

## Contributing a change

```sh
git fetch upstream
git checkout -b my-change upstream/main
# edit, test, commit
git push -u aylward my-change
```

Then open a pull request from the fork branch to `Project-MONAI/monai-physio-slicer`
`main`. Releases follow the same flow from a `release_<version>` branch; see
[SLICER_SUBMISSION.md](SLICER_SUBMISSION.md#versions-and-releases).

## Branches and tags upstream

- `main`: development; the Slicer Preview catalog entry builds it.
- `5.10`, `5.12`: the commits the Slicer 5.10 and 5.12 Stable catalog entries
  build. Move them only to commits validated in that Slicer release.
- Release tags: the bare version, for example `2026.11.1`.
- `Data`: a tag whose GitHub release hosts the sample data and the
  torch-scatter wheels used by setup. Keep it; the modules download from it.

## Repository settings

Slicer's Tier 1 checklist asks the upstream repository to have:

- the topic `3d-slicer-extension`;
- the description `3D Slicer extension for simulating physiological motion and
  exporting to Omniverse (OpenUSD) with MONAI-Physio.` or similar;
- the repository URL as its homepage;
- Wiki, Projects and Discussions disabled when unused. Keep Releases visible
  (they host the `Data` assets).

If the repository is renamed or moved, update the README links, the homepage,
icon and screenshot URLs in [CMakeLists.txt](../CMakeLists.txt), `RELEASE_URL`
in [runtime.py](../MONAIPhysioExportUSD/MONAIPhysioLib/runtime.py), and
`scm_url` in [ExtensionsIndex/MONAIPhysio.json](../ExtensionsIndex/MONAIPhysio.json)
and in the ExtensionsIndex entries.
