# tracers/pyNeuTube/neuron_structures.py

"""
neuron_structures.py

Intermediate neuronal structures. 
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


from .config import Defaults
from .stack_graph_utils import GraphEdge, Graph, GraphType
from .tracing import SegmentChains


@dataclass
class Neurocomp_Conn:
    """
    The meaning of info depends on the connection mode:
      - mode = NEUROCOMP_CONN_NONE: not connected. info has no meaning.
      - mode = NEUROCOMP_CONN_HL: info[0] is the position of the hook. info[1] is
        the position of the loop. hook position can only be the end index
        (0 for head and 1 for tail).
      - mode = NEUROCOMP_CONN_LINK: info[0] is the end index of the first
        component and info[1] is the end index of the second component.
    
    cost is the connection cost, which has no predefined lower or upper bound.
    """
    mode: int
    info: np.ndarray = field(metadata={"length": 2})
    pos: np.ndarray = field(metadata={"length": 3})
    ort: np.ndarray = field(metadata={"length": 3})
    cost: float
    min_pdist: float
    min_sdist: float


@dataclass
class NeuronStructure:
    """
    This data structure specifies the graph structure of neuron components. It has
    three parts, <graph>, <conn> and <comp>. <comp> is an SegmentChains object
    in the structure. <graph> specifies the connection relationship among the
    chains. <conn> specifies how the chains are connected. Each element of
    <conn> corresponds to each edge in <graph> with the same indexing rule. 
    """
    graph: Optional[Graph] = None
    conn: Optional[List[Neurocomp_Conn]] = None
    comp: Optional[SegmentChains] = None
 
def initialize_graph(nvertex: int, edge_capacity: int, weighted: bool):
    assert(edge_capacity <= Defaults.Max_Edge_Capacity)

    graph = Graph(directed=False, gtype=GraphType.GENERAL_GRAPH, weighted=weighted)
    #graph.nvertex = nvertex    # LYF: should be zero?
    # graph.edge_capacity = edge_capacity
    # graph.edges = [GraphEdge() for _ in range(edge_capacity)] #np.zeros((edge_capacity, 2), dtype=int)
    # if weighted:
    #     graph.weights = [0 for _ in range(edge_capacity)] #np.zeros((edge_capacity), dtype=float)
    #graph.edges = []
    #if weighted:
    #    graph.weights = []
    
    return graph
   
def neuron_structure_from_chains(chains: SegmentChains, zscale: float = 1.0):
    # Initialize the graph
    ns = NeuronStructure()
    ns.comp = chains
    ns.graph = initialize_graph(len(chains), 0, False)
    ns.conn = []
    return ns
    


class Circle:
    def __init__(self, center: np.ndarray, radius: float, theta: float = 0.0, psi: float = 0.0):
        self.center = center
        self.radius = radius
        self.theta = theta
        self.psi = psi
