"""Public tracing API and high-level pipeline helpers."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from pyneutube.core.io.image_parser import ImageParser
from pyneutube.core.io.swc_parser import Neuron
from pyneutube.core.processing.filtering import (
    connectivity_filter,
    local_max_filter,
    refine_local_max_threshold,
    subtract_background,
    threshold_filter,
    triangle_threshold,
)
from pyneutube.tracers.pyNeuTube.chains_to_morphology import ChainConnector
from pyneutube.tracers.pyNeuTube.seeds import Seed, Seeds
from pyneutube.tracers.pyNeuTube.tracing import SegmentChain, SegmentChains, TracingSegment

SUPPORTED_IMAGE_SUFFIXES = (
    ".tif",
    ".tiff",
    ".v3draw",
    ".raw",
    ".v3dpbd",
    ".pbd",
    ".h5",
    ".hdf5",
    ".nii",
    ".nii.gz",
    ".nrrd",
    ".nhdr",
)


@dataclass
class TracingResult:
    image_path: Path | None
    threshold: float
    seeds: Seeds
    chains: SegmentChains
    neuron: Neuron | None = None
    output_swc: Path | None = None
    signal_image: np.ndarray | None = None
    binary_image: np.ndarray | None = None
    skipped: bool = False
    skip_reason: str | None = None


def _vprint(verbose: int, message: str) -> None:
    if verbose:
        print(message)


def _time_step(verbose: int, label: str, started_at: float) -> None:
    if verbose:
        print(f"{label}: {perf_counter() - started_at:.3f}s")


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if n_jobs <= 0:
        raise ValueError("`n_jobs` must be a positive integer or -1.")
    return n_jobs


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _overwrite_error(path: Path, *, mode: str) -> FileExistsError:
    if mode == "single":
        return FileExistsError(f"Output already exists: {path}. Pass overwrite=True to replace it.")
    return FileExistsError(
        f"Output already exists: {path}. Pass overwrite=True or --overwrite to replace it."
    )


def _resolve_on_exists(on_exists: str | None, *, default: str) -> str:
    policy = default if on_exists is None else str(on_exists).lower()
    if policy not in {"error", "skip"}:
        raise ValueError("`on_exists` must be one of {'error', 'skip'}.")
    return policy


def _skipped_trace_result(
    image_path: Path,
    *,
    output_swc: Path | None = None,
    reason: str,
) -> TracingResult:
    return TracingResult(
        image_path=image_path,
        threshold=float("nan"),
        seeds=Seeds(),
        chains=SegmentChains(),
        neuron=None,
        output_swc=output_swc,
        skipped=True,
        skip_reason=reason,
    )


def _chain_coords(chains: SegmentChains) -> np.ndarray:
    coords = [chain.to_coords() for chain in chains if len(chain) > 0]
    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(coords, axis=0)


def _matches_suffix(path: Path, suffixes: Sequence[str]) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix.lower()) for suffix in suffixes)


def preprocess_volume(
    image: np.ndarray,
    *,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    signal_image = np.ascontiguousarray(np.asarray(image), dtype=np.float64)

    t0 = perf_counter()
    signal_image = subtract_background(signal_image, verbose=max(verbose - 1, 0))
    _time_step(verbose, "subtract_background", t0)

    t0 = perf_counter()
    local_max_mask = local_max_filter(signal_image)
    _time_step(verbose, "local_max_filter", t0)

    local_max_values = signal_image[local_max_mask > 0]
    if local_max_values.size == 0:
        raise ValueError("No local maxima were found in the input volume.")

    if np.ptp(local_max_values) == 0:
        threshold = max(float(local_max_values[0]) - 1.0, 0.0)
        _vprint(
            verbose,
            f"triangle_threshold skipped; using constant-value fallback {threshold:.3f}",
        )
    else:
        t0 = perf_counter()
        threshold = float(triangle_threshold(local_max_values))
        _time_step(verbose, f"triangle_threshold={threshold:.3f}", t0)

        t0 = perf_counter()
        threshold = float(refine_local_max_threshold(signal_image, threshold))
        _time_step(verbose, f"refine_local_max_threshold={threshold:.3f}", t0)

    t0 = perf_counter()
    binary_image = threshold_filter(signal_image, threshold)
    binary_image = connectivity_filter(binary_image, 4, n_neighbors=26)
    binary_image = binary_dilation(binary_image, structure=np.ones((3, 3, 3)), border_value=0)
    binary_image = binary_erosion(binary_image, structure=np.ones((3, 3, 3)), border_value=1)
    _time_step(verbose, "binary_mask", t0)

    return signal_image, np.ascontiguousarray(binary_image, dtype=np.uint8), threshold


def trace_volume(
    image: np.ndarray,
    *,
    n_jobs: int = 1,
    verbose: int = 1,
    return_intermediates: bool = False,
) -> TracingResult:
    return _trace_volume_internal(
        image,
        n_jobs=n_jobs,
        verbose=verbose,
        return_intermediates=return_intermediates,
    )


def _trace_volume_internal(
    image: np.ndarray,
    *,
    n_jobs: int = 1,
    verbose: int = 1,
    max_seeds: int | None = None,
    connect_chains: bool = True,
    filter_chains: bool = True,
    debug_dir: str | Path = "debug",
    return_intermediates: bool = False,
) -> TracingResult:
    resolved_n_jobs = _resolve_n_jobs(n_jobs)
    signal_image, binary_image, threshold = preprocess_volume(image, verbose=verbose)

    t0 = perf_counter()
    seeds = Seeds()
    seeds.generate_tracing_seeds(
        signal_image,
        binary_image,
        n_jobs=resolved_n_jobs,
        verbose=verbose,
    )
    _time_step(verbose, "generate_tracing_seeds", t0)

    t0 = perf_counter()
    chains = SegmentChains(image_shape=signal_image.shape)
    chains.generate_neuron_trace(seeds, signal_image, max_seeds=max_seeds, verbose=verbose)
    if filter_chains:
        chains.filter_chains(verbose=verbose)
    _time_step(verbose, "generate_neuron_trace", t0)

    neuron = None
    if connect_chains:
        t0 = perf_counter()
        connector = ChainConnector(debug=verbose >= 2, debug_dir=str(debug_dir), verbose=verbose)
        neuron = connector.reconstruct(chains, signal_image)
        _time_step(verbose, "reconstruct", t0)

    return TracingResult(
        image_path=None,
        threshold=threshold,
        seeds=seeds,
        chains=chains,
        neuron=neuron,
        signal_image=signal_image if return_intermediates else None,
        binary_image=binary_image if return_intermediates else None,
    )


def trace_file(
    image_path: str | Path,
    *,
    dataset: str | None = None,
    output_swc: str | Path | None = None,
    output_overlay: str | Path | None = None,
    n_jobs: int = 1,
    verbose: int = 1,
    overwrite: bool = False,
    on_exists: str | None = None,
    return_intermediates: bool = False,
) -> TracingResult:
    return _trace_file_internal(
        image_path,
        dataset=dataset,
        output_swc=output_swc,
        output_overlay=output_overlay,
        n_jobs=n_jobs,
        verbose=verbose,
        overwrite=overwrite,
        on_exists=on_exists,
        return_intermediates=return_intermediates,
    )


def _trace_file_internal(
    image_path: str | Path,
    *,
    dataset: str | None = None,
    output_swc: str | Path | None = None,
    output_overlay: str | Path | None = None,
    n_jobs: int = 1,
    verbose: int = 1,
    overwrite: bool = False,
    on_exists: str | None = None,
    max_seeds: int | None = None,
    connect_chains: bool = True,
    filter_chains: bool = True,
    debug_dir: str | Path = "debug",
    return_intermediates: bool = False,
) -> TracingResult:
    image_path = Path(image_path)
    output_swc_path = Path(output_swc) if output_swc is not None else None
    output_overlay_path = Path(output_overlay) if output_overlay is not None else None
    existing_output = None
    if output_swc_path is not None and output_swc_path.exists():
        existing_output = output_swc_path
    elif output_overlay_path is not None and output_overlay_path.exists():
        existing_output = output_overlay_path

    if existing_output is not None and not overwrite:
        exists_policy = _resolve_on_exists(on_exists, default="error")
        if exists_policy == "error":
            raise _overwrite_error(existing_output, mode="single")
        _vprint(
            verbose,
            f"Skipped {image_path.name} (exists; use overwrite=True to replace)",
        )
        return _skipped_trace_result(
            image_path,
            output_swc=output_swc_path,
            reason="exists",
        )

    parser = ImageParser(image_path, dataset=dataset, verbose=verbose)

    t0 = perf_counter()
    image = parser.load()
    _time_step(verbose, "load_image", t0)

    result = _trace_volume_internal(
        image,
        n_jobs=n_jobs,
        verbose=verbose,
        max_seeds=max_seeds,
        connect_chains=connect_chains,
        filter_chains=filter_chains,
        debug_dir=debug_dir,
        return_intermediates=return_intermediates,
    )
    result.image_path = image_path

    if output_swc is not None:
        if result.neuron is None:
            raise ValueError("No neuron reconstruction is available for SWC export.")
        result.output_swc = output_swc_path
        result.neuron.save_swc(result.output_swc, verbose=verbose)

    if output_overlay is not None:
        from pyneutube.visualization import save_overlay_figure

        overlay_coords = (
            result.neuron.coords if result.neuron is not None else _chain_coords(result.chains)
        )
        if overlay_coords.size == 0:
            raise ValueError("No coordinates are available for overlay export.")
        save_overlay_figure(image, overlay_coords, output_overlay_path, title=image_path.name)
        _vprint(verbose, f"Overlay saved to {output_overlay}")

    return result


def _trace_file_worker(
    payload: tuple[str, str, int, int],
) -> dict[str, object]:
    input_path, output_swc, n_jobs, verbose = payload
    started_at = perf_counter()
    try:
        result = trace_file(
            input_path,
            output_swc=output_swc,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    except Exception as exc:
        return {
            "timestamp_utc": _iso_timestamp(),
            "status": "failed",
            "input_path": input_path,
            "output_swc": output_swc,
            "elapsed_seconds": perf_counter() - started_at,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    return {
        "timestamp_utc": _iso_timestamp(),
        "status": "completed",
        "input_path": input_path,
        "output_swc": str(result.output_swc) if result.output_swc is not None else None,
        "elapsed_seconds": perf_counter() - started_at,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }


def _append_manifest_record(
    manifest_path: Path | None,
    record: dict[str, object],
) -> None:
    if manifest_path is None:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _skip_existing_record(input_path: Path, output_swc: Path) -> dict[str, object]:
    return {
        "timestamp_utc": _iso_timestamp(),
        "status": "skipped",
        "input_path": str(input_path),
        "output_swc": str(output_swc),
        "elapsed_seconds": 0.0,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "reason": "exists",
    }


def trace_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    batch_n_jobs: int = 1,
    trace_n_jobs: int = 1,
    verbose: int = 1,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
    on_exists: str | None = None,
) -> list[Path]:
    image_paths = [Path(path) for path in input_paths]
    if not image_paths:
        raise FileNotFoundError("No input image files were provided.")

    missing_paths = [path for path in image_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Input image file does not exist: {missing_paths[0]}")

    unsupported_paths = [path for path in image_paths if not _matches_suffix(path, SUPPORTED_IMAGE_SUFFIXES)]
    if unsupported_paths:
        raise ValueError(f"Unsupported image file suffix: {unsupported_paths[0]}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_batch_n_jobs = _resolve_n_jobs(batch_n_jobs)
    resolved_trace_n_jobs = _resolve_n_jobs(trace_n_jobs)
    exists_policy = _resolve_on_exists(on_exists, default="skip")
    manifest = Path(manifest_path) if manifest_path is not None else None

    _vprint(
        verbose,
        (
            f"Tracing {len(image_paths)} image(s) "
            f"with batch_n_jobs={resolved_batch_n_jobs}, trace_n_jobs={resolved_trace_n_jobs}"
        ),
    )

    completed_outputs: list[Path] = []
    jobs: list[tuple[str, str, int, int]] = []
    for image_path in image_paths:
        output_swc = output_dir / f"{image_path.name}.swc"
        if output_swc.exists() and not overwrite:
            if exists_policy == "error":
                raise _overwrite_error(output_swc, mode="batch")
            completed_outputs.append(output_swc)
            _append_manifest_record(manifest, _skip_existing_record(image_path, output_swc))
            _vprint(
                verbose,
                f"Skipped {image_path.name} -> {output_swc.name} (exists; use overwrite=True to replace)",
            )
            continue
        jobs.append(
            (
                str(image_path),
                str(output_swc),
                resolved_trace_n_jobs,
                max(verbose - 1, 0),
            )
        )

    def handle_record(record: dict[str, object]) -> None:
        status = str(record["status"])
        input_path = Path(str(record["input_path"]))
        output_swc_value = record["output_swc"]
        output_swc = Path(str(output_swc_value)) if isinstance(output_swc_value, str) else None
        _append_manifest_record(manifest, record)
        if status == "completed":
            if output_swc is not None:
                completed_outputs.append(output_swc)
                _vprint(verbose, f"Completed {input_path.name} -> {output_swc.name}")
            else:
                _vprint(verbose, f"Completed {input_path.name}")
            return
        if status == "skipped":
            if output_swc is not None:
                _vprint(verbose, f"Skipped {input_path.name} -> {output_swc.name}")
            else:
                _vprint(verbose, f"Skipped {input_path.name}")
            return

        _vprint(
            max(verbose, 1),
            f"Failed {input_path.name}: {record['error_type']}: {record['error_message']}",
        )

    if not jobs:
        return sorted(set(completed_outputs))

    if resolved_batch_n_jobs == 1:
        for job in jobs:
            handle_record(_trace_file_worker(job))
        return sorted(set(completed_outputs))

    with ProcessPoolExecutor(max_workers=min(resolved_batch_n_jobs, len(jobs))) as executor:
        futures = {executor.submit(_trace_file_worker, job): Path(job[0]) for job in jobs}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                input_path = futures[future]
                record = {
                    "timestamp_utc": _iso_timestamp(),
                    "status": "failed",
                    "input_path": str(input_path),
                    "output_swc": str(output_dir / f"{input_path.name}.swc"),
                    "elapsed_seconds": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }

            handle_record(record)

    return sorted(set(completed_outputs))


def trace_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    suffixes: Sequence[str] = SUPPORTED_IMAGE_SUFFIXES,
    batch_n_jobs: int = 1,
    trace_n_jobs: int = 1,
    verbose: int = 1,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
    on_exists: str | None = None,
) -> list[Path]:
    input_dir = Path(input_dir)
    image_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and _matches_suffix(path, suffixes)
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files were found in {input_dir}.")

    return trace_files(
        image_paths,
        output_dir,
        batch_n_jobs=batch_n_jobs,
        trace_n_jobs=trace_n_jobs,
        verbose=verbose,
        manifest_path=manifest_path,
        overwrite=overwrite,
        on_exists=on_exists,
    )


__all__ = [
    "Seed",
    "Seeds",
    "TracingSegment",
    "SegmentChain",
    "SegmentChains",
    "TracingResult",
    "SUPPORTED_IMAGE_SUFFIXES",
    "preprocess_volume",
    "trace_volume",
    "trace_file",
    "trace_files",
    "trace_directory",
]


