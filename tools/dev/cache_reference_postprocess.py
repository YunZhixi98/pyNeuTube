"""Create a reusable cache for reconstruction post-processing benchmarks."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyneutube.tracing as tracing_api
from pyneutube.core.io.swc_parser import Neuron
from pyneutube.tracers.pyNeuTube.chains_to_morphology import ChainConnector, postprocess_reconstruction


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_path",
        nargs="?",
        default="examples/data/reference_volume.nii.gz",
        help="Input image path.",
    )
    parser.add_argument(
        "--cache-path",
        default="tools/dev/reference_postprocess_cache.pkl",
        help="Where to write the pickle cache.",
    )
    parser.add_argument("--max-seeds", type=int, default=None, help="Optional tracing seed cap.")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Load the cache, run only post-processing, and print the final node count.",
    )
    return parser.parse_args(argv)


def _build_cache(image_path: Path, *, max_seeds: int | None) -> dict[str, object]:
    result = tracing_api._trace_file_internal(
        image_path,
        n_jobs=1,
        verbose=0,
        max_seeds=max_seeds,
        connect_chains=False,
        return_intermediates=True,
    )
    connector = ChainConnector(verbose=0)
    connector.prepare_chain_conn(result.chains, result.signal_image)
    connector.remove_redundant_edges()
    connector.crossover_test(result.chains)
    circle_graph, circle_conn_list, circle_comp_list = connector.chains_to_circles(result.chains)
    circle_graph.weights = [circle_conn_list[i].cost for i in range(circle_graph.nedge)]

    graph = nx.Graph()
    graph.add_nodes_from(range(len(circle_comp_list)))
    for i, (u, v) in enumerate(circle_graph.edges):
        graph.add_edge(u, v, weight=circle_graph.weights[i])
    tree = nx.minimum_spanning_tree(graph, weight="weight")

    neuron = Neuron().from_graph(tree, circle_comp_list)
    return {
        "image_path": str(image_path),
        "max_seeds": max_seeds,
        "swc_after_from_graph": neuron.swc.copy(),
    }


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(args.cache_path)
    if args.replay:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        neuron = Neuron().initialize(payload["swc_after_from_graph"].copy())
        postprocess_reconstruction(neuron, verbose=1)
        print(f"Final node count: {len(neuron)}")
        return

    image_path = Path(args.image_path)
    payload = _build_cache(image_path, max_seeds=args.max_seeds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(cache_path)


if __name__ == "__main__":
    main()
