"""Create a lightweight overlay from the bundled lightweight reference volume and SWC."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyneutube import ImageParser, Neuron, save_overlay_figure


def main() -> Path:
    image_path = REPO_ROOT / "examples" / "data" / "reference_volume_lite.nii.gz"
    swc_path = REPO_ROOT / "examples" / "data" / "reference_neutube.swc"
    output_path = REPO_ROOT / "examples" / "reference_overlay.png"

    image = ImageParser(image_path).load()
    neuron = Neuron().initialize(swc_path)
    save_overlay_figure(image, neuron.coords, output_path, title="Reference NeuTube reconstruction")
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
