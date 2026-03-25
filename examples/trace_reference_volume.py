"""Run a conservative end-to-end tracing smoke test on the bundled reference volume."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyneutube import trace_file


def main():
    image_path = REPO_ROOT / "examples" / "data" / "reference_volume_lite.nii.gz"
    output_swc = REPO_ROOT / "examples" / "reference_trace_smoke.swc"
    output_overlay = REPO_ROOT / "examples" / "reference_trace_smoke.png"

    result = trace_file(
        image_path,
        output_swc=output_swc,
        output_overlay=output_overlay,
        n_jobs=1,
        verbose=1,
        overwrite=True,
    )
    print(output_swc)
    print(output_overlay)
    return result


if __name__ == "__main__":
    main()
