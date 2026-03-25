## `pyneutube.tracers.pyNeuTube`

Internal NeuTube-style tracing implementation.

This package contains:

- seed generation
- tracing segments and segment chains
- graph-based chain connection
- geometry helpers and Cython acceleration modules

This module tree is not part of the stable public API. Release consumers should use `pyneutube.trace_volume`, `pyneutube.trace_file`, `pyneutube.trace_files`, or `pyneutube.trace_directory`.
