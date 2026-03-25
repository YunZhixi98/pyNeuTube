## `pyneutube.core`

Internal shared building blocks for the public package surface.

Current contents:

- `io`: volume and SWC parsing helpers
- `processing`: preprocessing, segmentation, morphology, and Cython-backed kernels
- `math_utils` and `neighbors`: shared numerical utilities used by the tracing pipeline

This package is internal. Downstream code should prefer the top-level `pyneutube` API unless it is intentionally working against implementation details.
