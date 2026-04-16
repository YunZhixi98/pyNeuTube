import argparse
from pathlib import Path

from pyneutube.tracing import trace_directory, trace_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Trace one image or a directory of images into SWC files.",
    )
    parser.add_argument("input_path", help="Input image file or directory.")
    parser.add_argument("--output-swc", help="SWC output path for a single image.")
    parser.add_argument(
        "--visualization-dir",
        help=(
            "Optional directory for lightweight PNG overlays. "
            "Outputs are written to result/, seeds/, chains/, and pre_postprocess/. Disabled by default."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional Python module path for trace config overrides. "
            "Defaults to the built-in tracer config."
        ),
    )
    parser.add_argument("--output-dir", help="Output directory for batch SWC files.")
    parser.add_argument("--manifest-path", help="Optional JSONL run log for batch mode.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="CPU workers for tracing one image.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional per-image timeout in seconds. Disabled by default.",
    )
    parser.add_argument(
        "--batch-n-jobs",
        type=int,
        default=1,
        help="How many files to process at once in batch mode.",
    )
    parser.add_argument(
        "--trace-n-jobs",
        type=int,
        default=1,
        help="CPU workers used inside each batch job.",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0=silent).")
    parser.add_argument(
        "--on-exists",
        choices=("error", "skip"),
        default=None,
        help=(
            "What to do when overwrite=False and the output already exists. "
            "Default: error for single files, skip for batch mode."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs. If set, this overrides --on-exists.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input_path)

    if input_path.is_dir():
        if args.output_swc:
            raise ValueError("`--output-swc` is only supported for single-image tracing.")
        if args.output_dir is None:
            raise ValueError("Batch tracing requires `--output-dir`.")

        outputs = trace_directory(
            input_path,
            args.output_dir,
            visualization_dir=args.visualization_dir,
            batch_n_jobs=args.batch_n_jobs,
            trace_n_jobs=args.trace_n_jobs,
            trace_timeout=args.timeout,
            verbose=args.verbose,
            manifest_path=args.manifest_path,
            overwrite=args.overwrite,
            on_exists=args.on_exists,
            config=args.config,
        )
        if args.verbose:
            for output in outputs:
                print(output)
        return outputs

    output_swc = (
        Path(args.output_swc) if args.output_swc else input_path.with_name(f"{input_path.name}.swc")
    )
    result = trace_file(
        input_path,
        output_swc=output_swc,
        visualization_dir=args.visualization_dir,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        verbose=args.verbose,
        overwrite=args.overwrite,
        on_exists=args.on_exists,
        config=args.config,
    )
    if args.verbose and result.output_swc is not None:
        print(result.output_swc)
    return result


if __name__ == "__main__":
    main()
