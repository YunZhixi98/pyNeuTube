# Tutorial

## 1. Install

Compile the default acceleration layer and install the package:

```bash
python -m pip install -e .[dev,test,io,visualization]
```

If you modify `.pyx` files, refresh tracked generated C sources before release:

```bash
python scripts/regenerate_cython_sources.py
```

## 2. Load an image volume

```python
from pyneutube import ImageParser

image = ImageParser("examples/data/reference_volume_lite.nii.gz").load()
print(image.shape)
```

Supported formats:

- default: TIFF, Vaa3D raw, Vaa3D PBD
- default also includes: NIfTI
- with `pyneutube[io]`: HDF5 and NRRD

Saving uses the output suffix to choose a format. `dataset` and `compression` only affect HDF5 saves, and `affine` only affects NIfTI saves. Example:

```python
from pyneutube import ImageParser

ImageParser.save(image, "volume.h5", dataset="/volume", compression="gzip")
ImageParser.save(image, "volume.nii.gz")
```

## 3. Run a conservative trace

```python
from pyneutube import trace_file

result = trace_file(
    "examples/data/reference_volume_lite.nii.gz",
    output_swc="reference_trace.swc",
    output_overlay="reference_trace.png",
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

## 4. Preprocess without full tracing

```python
from pyneutube import ImageParser, preprocess_volume

image = ImageParser("examples/data/reference_volume_lite.nii.gz").load()
signal, binary_mask, threshold = preprocess_volume(image, verbose=1)
```

## 5. Batch processing

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

Batch tracing separates outer file-level parallelism (`batch_n_jobs`) from per-file tracing parallelism (`trace_n_jobs`). With `overwrite=False`, existing SWC outputs are skipped. The optional manifest file records completed, failed, and skipped entries in JSONL format.

## 6. Example scripts

- `python -m examples.overlay_reference_reconstruction`
- `python -m examples.trace_reference_volume`
- `python scripts/convert_reference_formats.py`
- `python scripts/profile_reference_pipeline.py --max-seeds 2`

