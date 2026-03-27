"""Run a conservative end-to-end tracing smoke test on the bundled reference volume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyneutube import save_overlay_figure, trace_file


def main():
    parser = argparse.ArgumentParser(description="Trace a reference volume")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for parallel processing")
    args = parser.parse_args()
    
    image_path = REPO_ROOT / "examples" / "data" / "reference_volume.nii.gz"
    output_swc = REPO_ROOT / "examples" / "reference_volume_trace.swc"
    output_overlay = REPO_ROOT / "examples" / "reference_volume_trace.png"

    result = trace_file(
        image_path,
        output_swc=output_swc,
        n_jobs=args.n_jobs,
        verbose=1,
        overwrite=True,
    )
    if result.neuron is not None:
        save_overlay_figure(image_path, result.neuron, output_overlay, title=image_path.name)
    print(output_swc)
    print(output_overlay)
    return result


if __name__ == "__main__":
    main()
