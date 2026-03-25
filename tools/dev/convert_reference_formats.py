"""Convert the bundled reference volume into other supported formats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert the bundled reference volume into supported formats.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="examples/data/reference_volume_lite.nii.gz",
        help="Source image path. Defaults to the bundled NIfTI example volume.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/converted",
        help="Directory for converted outputs.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    from pyneutube import ImageParser

    args = parse_args(argv)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = input_path.stem

    targets = [
        output_dir / f"{stem}.v3draw",
        output_dir / f"{stem}.h5",
        output_dir / f"{stem}.nii.gz",
    ]

    try:
        import nrrd  # noqa: F401
    except ImportError:
        pass
    else:
        targets.append(output_dir / f"{stem}.nrrd")

    for target in targets:
        ImageParser.convert(input_path, target, overwrite=True)
        print(target)

    return targets


if __name__ == "__main__":
    main()
