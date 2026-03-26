# Tutorial

## 1. Install

Install the package with development and optional I/O dependencies:

```bash
python -m pip install -e .[dev,test,io]
```

If you modify `.pyx` files, regenerate the tracked C sources before release:

```bash
python tools/dev/regenerate_cython_sources.py
```

## 2. Load an image volume

```python
from pyneutube import ImageParser

image = ImageParser("examples/data/reference_volume.nii.gz").load()
print(image.shape)
```

Supported input formats:

- default: TIFF, Vaa3D raw, Vaa3D PBD, NIfTI
- with `pyneutube[io]`: HDF5 and NRRD

Saving uses the output suffix to choose a format:

```python
from pyneutube import ImageParser

ImageParser.save(image, "volume.h5", dataset="/volume", compression="gzip")
ImageParser.save(image, "volume.nii.gz")
```

## 3. Run a trace

```python
from pyneutube import trace_file

result = trace_file(
    "examples/data/reference_volume.nii.gz",
    output_swc="reference_trace.swc",
    visualization_dir="visualizations",
    n_jobs=1,
    verbose=1,
    overwrite=False,
)

print(result.threshold, len(result.seeds), len(result.chains))
```

Key runtime controls:

- `n_jobs`: per-volume parallelism for single-image tracing
- `batch_n_jobs`: file-level batch parallelism
- `trace_n_jobs`: per-file parallelism inside each batch worker
- `overwrite`: replace existing outputs
- `verbose`: controls progress logging
- `visualization_dir`: optional output directory for lightweight PNG overlays

The public tracing entry points intentionally keep this parameter set small. They are designed for stable file- and runtime-level control, not for exposing every internal tracing heuristic. Lower-level tracing constants and reconstruction rules still live in internal modules and may change between revisions, so method-tuning experiments should pin a specific commit and document any internal overrides explicitly.

## 4. Preprocess without full tracing

```python
from pyneutube import ImageParser, preprocess_volume

image = ImageParser("examples/data/reference_volume.nii.gz").load()
signal, binary_mask, threshold = preprocess_volume(image, verbose=1)
```

## 5. Batch processing

```python
from pyneutube import trace_directory

outputs = trace_directory(
    "input_volumes",
    "swc_outputs",
    visualization_dir="visualizations",
    batch_n_jobs=4,
    trace_n_jobs=1,
    verbose=1,
    manifest_path="trace_manifest.jsonl",
    overwrite=False,
)
```

Batch tracing separates outer file-level parallelism from per-file tracing parallelism. With `overwrite=False`, existing SWC outputs are skipped. The optional manifest file records completed, failed, and skipped entries in JSONL format.

## 6. Manual visualization

```python
from pyneutube import save_overlay_figure

save_overlay_figure(
    "examples/data/reference_volume.nii.gz",
    "examples/data/reference_neutube.swc",
    "reference_overlay.png",
)
```

`image` can be either a path or a loaded volume. `trace` can be an SWC path, a `Neuron`, or an `(N, 3)` coordinate array. The rendered background uses a `log1p`-transformed maximum-intensity projection and the axes are fixed to the image shape.

## 7. Local examples and tools

- `python -m examples.trace_reference_volume`
- `python -m examples.overlay_reference_reconstruction`
- `python tools/dev/smoke_imports.py`
- `python tools/dev/convert_reference_formats.py`
- `python tools/dev/profile_reference_pipeline.py --max-seeds 2`
