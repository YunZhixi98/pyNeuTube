# PyNeuTube

PyNeuTube is a Python reimplementation of the NeuTube tracing workflow for 3D microscopy volumes. This repository is being prepared as a reusable open-source package with three priorities:

- compiled, reproducible tracing performance through mandatory Cython-backed extensions,
- a small stable public API for scripting and downstream reuse,
- GitHub/PyPI-ready tests, examples, and release metadata.

## Current scope

PyNeuTube currently includes:

- multi-format 3D image I/O through `ImageParser`,
- SWC parsing/export through `Neuron`,
- NeuTube-style preprocessing, seed generation, tracing, and chain connection,
- lightweight overlay visualization helpers,
- a conservative single-file pipeline API plus explicit-file and directory-level batch entry points.

The default build path now compiles the acceleration layer. Pure Python fallbacks for the critical Cython modules are no longer shipped.

## Installation

Source install:

```bash
python -m pip install .
```

Editable install for development and testing:

```bash
python -m pip install -e .[dev,test,io,visualization]
```

Optional extras:

- `io`: HDF5 and NRRD support via `h5py` and `pynrrd`
- `visualization`: heavier plotting/geometry helpers
- `test`: pytest plus optional I/O test dependencies
- `dev`: build, lint, packaging, and Cython regeneration tools

Notes:

- The default package build compiles C extensions from the tracked generated C sources.
- Building from source therefore requires a working C/C++ toolchain unless you install a prebuilt wheel.
- `v0.1.0` is validated against both NumPy 1.26 and NumPy 2.2 on Python 3.10. Release builds should be produced against NumPy 2 headers, while runtime remains compatible with `numpy>=1.24`.
- The maintainer helper `python scripts/regenerate_cython_sources.py` refreshes tracked `.c` sources from `.pyx` files before release.

## Supported image formats

Default support:

- TIFF stacks: `.tif`, `.tiff`
- Vaa3D raw volumes: `.v3draw`, `.raw`
- Vaa3D PBD-compressed volumes: `.v3dpbd`, `.pbd`

Additional formats with `pyneutube[io]` installed:

- HDF5: `.h5`, `.hdf5`
- NRRD: `.nrrd`, `.nhdr`

Default support also includes:

- NIfTI: `.nii`, `.nii.gz`

Current output support:

- SWC export through `Neuron.save_swc`
- Image export to TIFF, Vaa3D raw, HDF5, NIfTI, and NRRD through `ImageParser.save`

Saving uses the output suffix to choose a format. `dataset` and `compression` only affect HDF5 saves, while `affine` only affects NIfTI saves.

```python
from pyneutube import ImageParser

ImageParser.save(image, "volume.h5", dataset="/volume", compression="gzip")
ImageParser.save(image, "volume.nii.gz")
```

## Quick start

Most users only need one of these four entry points. The bundled reference example below is a NIfTI volume and now works with the default install. `overwrite=True` always replaces existing outputs. If `overwrite=False`, `trace_file` defaults to `on_exists="error"` and `trace_directory` defaults to `on_exists="skip"`.

Single volume:

```python
from pyneutube import trace_file

result = trace_file(
    "examples/data/reference_volume_lite.nii.gz",
    output_swc="reference_trace.nii.gz.swc",
    n_jobs=1,
    verbose=1,
    overwrite=False,
)

print(result.threshold, len(result.seeds), len(result.chains))
```

Explicit file list batch:

```python
from pyneutube import trace_files

outputs = trace_files(
    ["sample_a.tif", "sample_b.tif", "sample_c.nii.gz"],
    "swc_outputs",
    batch_n_jobs=4,
    trace_n_jobs=1,
    overwrite=False,
)
```

Command line:

```bash
pyneutube-trace examples/data/reference_volume_lite.nii.gz --verbose 1
pyneutube-trace input_volumes --output-dir swc_outputs --batch-n-jobs 4 --trace-n-jobs 1
```

## Formats and Common Errors

| Topic | Value | Notes |
| --- | --- | --- |
| Input formats | `.tif`, `.tiff`, `.v3draw`, `.raw`, `.v3dpbd`, `.pbd` | Available in the base install. |
| Extra input formats | `.h5`, `.hdf5`, `.nrrd`, `.nhdr` | Require `pyneutube[io]`. |
| Default medical-image input | `.nii`, `.nii.gz` | Available in the base install via `nibabel`. |
| Tracing output | `.swc` | Produced by `trace_file`, `trace_files`, and `trace_directory`. |
| Overlay output | image path such as `.png` | Produced by `output_overlay`. |
| Save formats | TIFF, Vaa3D raw, HDF5, NIfTI, NRRD | Chosen from the output suffix in `ImageParser.save`. |
| Existing output | `overwrite=True` | Always replaces the old file. |
| Existing output with `overwrite=False` | `trace_file`: error; `trace_directory`: skip | Override with `on_exists="skip"` or `on_exists="error"`. |
| Missing optional I/O dependency | `ImportError` mentioning `h5py` or `pynrrd` | Install `pyneutube[io]` or the missing package directly. |
| Empty / unusable volume | `ValueError: No local maxima were found...` | Usually means the image has no traceable foreground after preprocessing. |

## Public API

For most users, the primary public API is:

- `trace_file`: trace one volume file and optionally write SWC / overlay output
- `trace_files`: batch trace an explicit list / tuple / 1D array of input paths
- `trace_directory`: collect supported files from a directory, then delegate to `trace_files`
- `trace_volume`: trace an in-memory NumPy volume when you already manage I/O yourself

Supporting public helpers are available when needed:

- `ImageParser`, `Neuron`
- `preprocess_volume`
- `subtract_background`, `threshold_filter`, `triangle_threshold`
- `refine_local_max_threshold`, `local_max_filter`, `connectivity_filter`
- `save_overlay_figure`

Everything under `pyneutube.core.*` and `pyneutube.tracers.*` should be treated as internal implementation detail unless explicitly documented otherwise. Low-level tracing types such as `Seed`, `TracingSegment`, and `SegmentChain` are intentionally no longer exported from the package root.

## Batch processing

Directory-level batch tracing is available through `trace_directory`, which is a convenience wrapper around `trace_files`:

```python
from pyneutube import trace_directory

outputs = trace_directory(
    "input_volumes",
    "swc_outputs",
    batch_n_jobs=4,
    trace_n_jobs=1,
    verbose=1,
    manifest_path="trace_manifest.jsonl",
    overwrite=False,
)
```

Batch tracing separates outer file-level parallelism (`batch_n_jobs`) from per-file tracing parallelism (`trace_n_jobs`) so batch runs do not multiply worker counts accidentally. When `overwrite=False`, existing SWC outputs are skipped. If `manifest_path` is provided, it writes JSONL records for completed, failed, and skipped files.

## Reference data and examples

Bundled examples include:

- `examples/data/reference_volume_lite.nii.gz`: lightweight compressed reference volume for demos and tests
- `examples/overlay_reference_reconstruction.py`: overlay the original NeuTube SWC on the reference volume
- `examples/trace_reference_volume.py`: conservative tracing/reconstruction smoke example
- `scripts/convert_reference_formats.py`: convert the bundled NIfTI reference into Vaa3D raw, HDF5, NIfTI, and optional NRRD example assets
- `scripts/profile_reference_pipeline.py`: cProfile helper for reference-volume hotspot analysis

## Development workflow

Recommended local checks:

```bash
python Cython_setup.py build_ext --inplace
python scripts/smoke_imports.py
python -m pytest -q
ruff check .
# Build from a clean tree so local build/ artifacts do not shadow the build frontend.
python scripts/profile_reference_pipeline.py --max-seeds 2
```

Before running `python -m build`, remove generated `build/`, `dist/`, and `pyneutube.egg-info/` directories and ensure the `build` frontend is installed in the active environment.

Conda recipe metadata is available under `conda-recipe/`, and a development environment file is provided at `environment.yml`.

## Limitations

- The project is still appropriate for a `v0.1.0` first public release, not a `v1.0.0` stability claim.
- Wider reconstruction regression against the original NeuTube implementation still needs more datasets than the bundled reference pair.
- Release builds are now validated in a NumPy 2 build environment and have been smoke-tested at runtime on both NumPy 1.26 and NumPy 2.2.
- Batch manifests improve resumability, but there is still no full experiment database or scheduler integration.









