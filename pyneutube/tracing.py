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
from typing import Callable

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
from pyneutube.visualization import save_overlay_figure

SUPPORTED_IMAGE_SUFFIXES = (
    ".tif",
    ".tiff",
    ".v3draw",
    ".raw",
    ".v3dpbd",
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
    output_visualization: Path | None = None
    signal_image: np.ndarray | None = None
    binary_image: np.ndarray | None = None
    skipped: bool = False
    skip_reason: str | None = None


class TraceTimeoutError(TimeoutError):
    def __init__(self, timeout_seconds: float, stage: str | None = None) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.stage = stage
        if stage:
            message = f"Tracing timed out after {self.timeout_seconds:g} s during {stage}."
        else:
            message = f"Tracing timed out after {self.timeout_seconds:g} s."
        super().__init__(message)


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


def _make_timeout_checker(
    timeout: float | None,
) -> tuple[float | None, Callable[[str | None], None]]:
    if timeout is None:
        return None, (lambda stage=None: None)

    timeout_seconds = float(timeout)
    if timeout_seconds <= 0:
        raise ValueError("`timeout` must be a positive number of seconds or None.")

    deadline = perf_counter() + timeout_seconds

    def _check_timeout(stage: str | None = None) -> None:
        if perf_counter() > deadline:
            raise TraceTimeoutError(timeout_seconds, stage)

    return timeout_seconds, _check_timeout


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


def _matches_suffix(path: Path, suffixes: Sequence[str]) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix.lower()) for suffix in suffixes)


def _visualization_output_path(
    image_path: Path,
    visualization_dir: str | Path | None,
) -> Path | None:
    if visualization_dir is None:
        return None
    return Path(visualization_dir) / f"{image_path.name}.png"


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
    timeout: float | None = None,
    verbose: int = 1,
    return_intermediates: bool = False,
) -> TracingResult:
    return _trace_volume_internal(
        image,
        n_jobs=n_jobs,
        timeout=timeout,
        verbose=verbose,
        return_intermediates=return_intermediates,
    )


def _trace_volume_internal(
    image: np.ndarray,
    *,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
    max_seeds: int | None = None,
    connect_chains: bool = True,
    filter_chains: bool = True,
    return_intermediates: bool = False,
) -> TracingResult:
    resolved_n_jobs = _resolve_n_jobs(n_jobs)
    _, check_timeout = _make_timeout_checker(timeout)

    check_timeout("preprocess_volume")
    signal_image, binary_image, threshold = preprocess_volume(image, verbose=verbose)
    check_timeout("preprocess_volume")

    t0 = perf_counter()
    seeds = Seeds()
    seeds.generate_tracing_seeds(
        signal_image,
        binary_image,
        n_jobs=resolved_n_jobs,
        verbose=verbose,
        check_timeout=check_timeout,
    )
    _time_step(verbose, "generate_tracing_seeds", t0)
    check_timeout("generate_tracing_seeds")

    t0 = perf_counter()
    chains = SegmentChains(image_shape=signal_image.shape)
    chains.generate_neuron_trace(
        seeds,
        signal_image,
        max_seeds=max_seeds,
        verbose=verbose,
        check_timeout=check_timeout,
    )
    if filter_chains:
        check_timeout("filter_chains")
        chains.filter_chains(verbose=verbose)
    _time_step(verbose, "generate_neuron_trace", t0)
    check_timeout("generate_neuron_trace")

    neuron = None
    if connect_chains:
        t0 = perf_counter()
        connector = ChainConnector(verbose=verbose)
        neuron = connector.reconstruct(chains, signal_image, check_timeout=check_timeout)
        _time_step(verbose, "reconstruct", t0)
        check_timeout("reconstruct")

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
    visualization_dir: str | Path | None = None,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
    overwrite: bool = False,
    on_exists: str | None = None,
    return_intermediates: bool = False,
) -> TracingResult:
    return _trace_file_internal(
        image_path,
        dataset=dataset,
        output_swc=output_swc,
        visualization_dir=visualization_dir,
        n_jobs=n_jobs,
        timeout=timeout,
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
    visualization_dir: str | Path | None = None,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
    overwrite: bool = False,
    on_exists: str | None = None,
    max_seeds: int | None = None,
    connect_chains: bool = True,
    filter_chains: bool = True,
    return_intermediates: bool = False,
) -> TracingResult:
    _, check_timeout = _make_timeout_checker(timeout)
    image_path = Path(image_path)
    output_swc_path = Path(output_swc) if output_swc is not None else None
    output_visualization_path = _visualization_output_path(image_path, visualization_dir)
    existing_output = None
    if output_swc_path is not None and output_swc_path.exists():
        existing_output = output_swc_path
    elif output_visualization_path is not None and output_visualization_path.exists():
        existing_output = output_visualization_path

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
    check_timeout("load_image")
    image = parser.load()
    _time_step(verbose, "load_image", t0)
    check_timeout("load_image")

    result = _trace_volume_internal(
        image,
        n_jobs=n_jobs,
        timeout=timeout,
        verbose=verbose,
        max_seeds=max_seeds,
        connect_chains=connect_chains,
        filter_chains=filter_chains,
        return_intermediates=return_intermediates,
    )
    result.image_path = image_path

    if output_swc is not None:
        if result.neuron is None:
            raise ValueError("No neuron reconstruction is available for SWC export.")
        result.output_swc = output_swc_path
        result.neuron.save_swc(result.output_swc, verbose=verbose)

    if output_visualization_path is not None:
        if result.neuron is not None:
            result.output_visualization = save_overlay_figure(
                image,
                result.neuron,
                output_visualization_path,
                title=image_path.name,
            )
        else:
            chain_coords = [chain.to_coords() for chain in result.chains if len(chain) > 0]
            if chain_coords:
                coords = np.concatenate(chain_coords, axis=0)
                result.output_visualization = save_overlay_figure(
                    image,
                    coords,
                    output_visualization_path,
                    title=image_path.name,
                )
            else:
                raise ValueError("No coordinates are available for visualization export.")

    return result


def _trace_file_worker(
    payload: tuple[str, str, str | None, int, float | None, int],
) -> dict[str, object]:
    input_path, output_swc, visualization_dir, n_jobs, timeout, verbose = payload
    started_at = perf_counter()
    try:
        result = trace_file(
            input_path,
            output_swc=output_swc,
            visualization_dir=visualization_dir,
            n_jobs=n_jobs,
            timeout=timeout,
            verbose=verbose,
        )
    except Exception as exc:
        timed_out = isinstance(exc, TraceTimeoutError)
        return {
            "timestamp_utc": _iso_timestamp(),
            "status": "timed_out" if timed_out else "failed",
            "input_path": input_path,
            "output_swc": output_swc,
            "elapsed_seconds": perf_counter() - started_at,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "timeout_seconds": exc.timeout_seconds if timed_out else None,
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
        "timeout_seconds": None,
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
        "timeout_seconds": None,
    }


def trace_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    visualization_dir: str | Path | None = None,
    batch_n_jobs: int = 1,
    trace_n_jobs: int = 1,
    trace_timeout: float | None = None,
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
    jobs: list[tuple[str, str, str | None, int, float | None, int]] = []
    for image_path in image_paths:
        output_swc = output_dir / f"{image_path.name}.swc"
        output_visualization = _visualization_output_path(image_path, visualization_dir)
        has_existing_output = output_swc.exists() or (
            output_visualization is not None and output_visualization.exists()
        )
        if has_existing_output and not overwrite:
            if exists_policy == "error":
                raise _overwrite_error(output_swc if output_swc.exists() else output_visualization, mode="batch")
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
                None if visualization_dir is None else str(visualization_dir),
                resolved_trace_n_jobs,
                trace_timeout,
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
        if status == "timed_out":
            timeout_seconds = record.get("timeout_seconds")
            timeout_suffix = (
                f" (timeout={float(timeout_seconds):g} s)"
                if isinstance(timeout_seconds, (int, float))
                else ""
            )
            _vprint(
                max(verbose, 1),
                f"Timed out {input_path.name}: {record['error_type']}: {record['error_message']}{timeout_suffix}",
            )
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
                    "timeout_seconds": None,
                }

            handle_record(record)

    return sorted(set(completed_outputs))


def trace_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    suffixes: Sequence[str] = SUPPORTED_IMAGE_SUFFIXES,
    visualization_dir: str | Path | None = None,
    batch_n_jobs: int = 1,
    trace_n_jobs: int = 1,
    trace_timeout: float | None = None,
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
        visualization_dir=visualization_dir,
        batch_n_jobs=batch_n_jobs,
        trace_n_jobs=trace_n_jobs,
        trace_timeout=trace_timeout,
        verbose=verbose,
        manifest_path=manifest_path,
        overwrite=overwrite,
        on_exists=on_exists,
    )


def extract_trace_seeds(
    signal_image: np.ndarray,
    binary_image: np.ndarray,
    *,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
) -> Seeds:
    _, check_timeout = _make_timeout_checker(timeout)
    seeds = Seeds()
    seeds.generate_tracing_seeds(
        signal_image,
        binary_image,
        n_jobs=_resolve_n_jobs(n_jobs),
        verbose=verbose,
        check_timeout=check_timeout,
    )
    return seeds


def generate_trace_chains(
    seeds: Seeds,
    signal_image: np.ndarray,
    *,
    max_seeds: int | None = None,
    filter_chains: bool = True,
    timeout: float | None = None,
    verbose: int = 1,
) -> SegmentChains:
    _, check_timeout = _make_timeout_checker(timeout)
    chains = SegmentChains(image_shape=signal_image.shape)
    chains.generate_neuron_trace(
        seeds,
        signal_image,
        max_seeds=max_seeds,
        verbose=verbose,
        check_timeout=check_timeout,
    )
    if filter_chains:
        check_timeout("filter_chains")
        chains.filter_chains(verbose=verbose)
    return chains


def connect_trace_chains(
    chains: SegmentChains,
    signal_image: np.ndarray,
    *,
    timeout: float | None = None,
    verbose: int = 1,
) -> Neuron:
    _, check_timeout = _make_timeout_checker(timeout)
    connector = ChainConnector(verbose=verbose)
    neuron = connector.reconstruct(chains, signal_image, check_timeout=check_timeout)
    if neuron is None:
        raise ValueError("No neuron reconstruction is available.")
    return neuron


__all__ = [
    "Seed",
    "Seeds",
    "TracingSegment",
    "SegmentChain",
    "SegmentChains",
    "TracingResult",
    "TraceTimeoutError",
    "SUPPORTED_IMAGE_SUFFIXES",
    "trace_file",
    "trace_volume",
    "trace_files",
    "trace_directory",
    "save_overlay_figure",
    "ImageParser",
    "Neuron",
    "preprocess_volume",
    "extract_trace_seeds",
    "generate_trace_chains",
    "connect_trace_chains",
]
