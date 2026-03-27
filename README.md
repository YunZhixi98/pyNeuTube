# PyNeuTube

PyNeuTube is a Python reimplementation of the NeuTube tracing workflow for 3D microscopy volumes. This repository is trimmed to a minimal reusable release surface focused on image I/O, preprocessing, tracing, and SWC export.

## Current scope

PyNeuTube currently provides:

- multi-format 3D image I/O through `ImageParser`
- SWC parsing and export through `Neuron`
- preprocessing, seed generation, tracing, and chain connection
- optional lightweight overlay visualization for tracing outputs
- public APIs for tracing a single file, multiple files, a directory, or an in-memory volume
- a single command-line entry point for SWC generation

Heavy debug inspection modules are intentionally not part of the release surface.

## Installation

Recommended Python versions: `3.10` to `3.12`.

Install from a local source tree with `pip`:

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

Install directly from GitHub source:

```bash
python -m pip install "git+https://github.com/YunZhixi98/pyNeuTube.git"
```

Create a `conda` environment from the bundled environment file:

```bash
conda env create -f environment.yml
conda activate pyneutube
```

Build and install a local `conda` package:

```bash
conda build conda-recipe
conda install --use-local pyneutube
```

These install paths include all supported runtime I/O formats.

## Supported formats

Default input support:

- TIFF stacks: `.tif`, `.tiff`
- Vaa3D raw volumes: `.v3draw`, `.raw`
- Vaa3D PBD-compressed volumes: `.v3dpbd`
- HDF5: `.h5`, `.hdf5`
- NIfTI: `.nii`, `.nii.gz`
- NRRD: `.nrrd`, `.nhdr`

Current output support:

- SWC export through `Neuron.save_swc`
- image export through `ImageParser.save`
- lightweight overlay export through `save_overlay_figure` or `visualization_dir`

## Quick start

Single volume:

```python
from pyneutube import trace_file

result = trace_file(
    "examples/data/reference_volume.nii.gz",
    output_swc="reference_trace.swc",
    visualization_dir="visualizations",
    n_jobs=1,
    timeout=600,
    verbose=1,
    overwrite=False,
)

print(result.threshold, len(result.seeds), len(result.chains))
```

Explicit file-list batch:

```python
from pyneutube import trace_files

outputs = trace_files(
    ["sample_a.tif", "sample_b.tif", "sample_c.nii.gz"],
    "swc_outputs",
    batch_n_jobs=4,
    trace_n_jobs=1,
    trace_timeout=600,
    overwrite=False,
)
```

Directory batch:

```python
from pyneutube import trace_directory

outputs = trace_directory(
    "input_volumes",
    "swc_outputs",
    batch_n_jobs=4,
    trace_n_jobs=1,
    trace_timeout=600,
    verbose=1,
    manifest_path="trace_manifest.jsonl",
    overwrite=False,
)
```

Command line:

```bash
pyneutube-trace examples/data/reference_volume.nii.gz --verbose 1
pyneutube-trace examples/data/reference_volume.nii.gz --timeout 600 --verbose 1
pyneutube-trace input_volumes --output-dir swc_outputs --visualization-dir visualizations --batch-n-jobs 4 --trace-n-jobs 1 --timeout 600
```

## Lightweight visualization

Visualization is optional and disabled by default during tracing. When `visualization_dir` is set, tracing writes a PNG maximum-intensity projection overlay for each processed image. The background image uses a `log1p`-transformed MIP and is clipped to the full image extent.

Manual use is also supported:

```python
from pyneutube import save_overlay_figure

save_overlay_figure(
    "examples/data/reference_volume.nii.gz",
    "examples/data/reference_neutube.swc",
    "reference_overlay.png",
)
```

`image` may be either an image path or an in-memory volume. `trace` may be an SWC path, a `Neuron`, or an `(N, 3)` coordinate array.

## Public API

The supported top-level API is:

- `trace_file`
- `trace_volume`
- `trace_files`
- `trace_directory`
- `save_overlay_figure`
- `ImageParser`
- `Neuron`

The staged public API is:

- `preprocess_volume`
- `extract_trace_seeds`
- `generate_trace_chains`
- `connect_trace_chains`

Additional public utilities:

- `SUPPORTED_IMAGE_SUFFIXES`
- `subtract_background`
- `threshold_filter`
- `triangle_threshold`
- `refine_local_max_threshold`
- `local_max_filter`
- `connectivity_filter`

Modules under `pyneutube.core.*` and `pyneutube.tracers.*` should be treated as internal implementation details unless explicitly documented otherwise.

The high-level tracing API is intentionally narrow: it exposes stable runtime controls such as I/O, parallelism, timeout, overwrite policy, and verbosity, but keeps most tracing heuristics internal. This keeps the release surface smaller and easier to maintain, at the cost of not exposing every low-level tracing knob through `trace_volume()` and `trace_file()`. For controlled experiments or method development, use the staged public API or inspect the internal tracer modules directly and pin the exact revision you evaluate.

## Examples and developer tools

Bundled examples and local release helpers:

- `python -m examples.trace_reference_volume`
- `python -m examples.overlay_reference_reconstruction`
- `python tools/dev/smoke_imports.py`
- `python tools/dev/regenerate_cython_sources.py`
- `python tools/dev/profile_reference_pipeline.py --max-seeds 2`
- `python tools/dev/convert_reference_formats.py`

## Development checks

Recommended local checks:

```bash
python Cython_setup.py build_ext --inplace
python tools/dev/smoke_imports.py
python -m pytest -q
ruff check .
```

## License

This project is distributed under the BSD 3-Clause License. See `LICENSE` for details.
