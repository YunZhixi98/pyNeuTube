"""Profile the reference tracing pipeline on a bundled example volume."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_path",
        nargs="?",
        default="examples/data/reference_volume.nii.gz",
        help="Input image path to profile.",
    )
    parser.add_argument("--max-seeds", type=int, default=2, help="Optional tracing seed cap.")
    parser.add_argument("--top", type=int, default=25, help="Number of profiling rows to print.")
    parser.add_argument(
        "--connect-chains",
        action="store_true",
        help="Profile the reconstruction stage in addition to tracing.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    import pyneutube.tracing as tracing_api

    args = parse_args(argv)
    image_path = Path(args.image_path)

    profiler = cProfile.Profile()
    profiler.enable()
    tracing_api._trace_file_internal(
        image_path,
        n_jobs=1,
        verbose=0,
        max_seeds=args.max_seeds,
        connect_chains=args.connect_chains,
        filter_chains=False,
    )
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer)
    stats.sort_stats("cumulative").print_stats(args.top)
    print(buffer.getvalue())


if __name__ == "__main__":
    main()
