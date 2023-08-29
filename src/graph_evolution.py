import argparse
import os
import random
import time
import numpy as np
import networkx as nx

# local imports
import graph_evolution_metrics as metrics

## CHANGE TO MATCH ALG AND DATASET
import plocal_fair_ppr as alg # addition alg
import selection.edge_selection as rm # removal alg

from selection.node_selection import SelectAll as Selector

# Graph Evolution
ITERATIONS = 30

# Metrics
VISIBILITY_RATIO = 0.1

# Constants for I/O
INPUT_DIR = "../input"
OUTPUT_DIR = "../data"


def add_edges(nx_g, edges):
    """
    Adds edges to given graph.
    """
    nx_g.add_edges_from(edges)
    if not DIRECTED:
        nx_g.add_edges_from([(v, u) for (u, v) in edges])


def remove_edges(nx_g, edges):
    """
    Removes edges from given graph.
    """
    nx_g.remove_edges_from(edges)
    if not DIRECTED:
        nx_g.remove_edges_from([(v, u) for (u, v) in edges])

    
# IMPORTANT: we do deletions to keep a baseline for the clustering coefficient and 
# see the effect of the algorithm on the graphs structure
def evolve_network(nx_g, minorities):
    """
    Iteratively evolve the network by adding and removing edges.
    """
    # initialize algorithm
    nx_g = alg.initialize(nx_g, directed=DIRECTED, protected=minorities)

    # initialize selection
    selector = Selector(nx_g, directed=DIRECTED, protected=minorities)

    # initialize recorder for metrics
    recorder = metrics.Recorder(directed=DIRECTED,
                            protected=minorities,
                            visibility_ratio=VISIBILITY_RATIO,
                            output_dir=OUTPUT_DIR,
                            output_prefix=OUTPUT_PREFIX)
    
    # initial metrics
    recorder.clear_files()
    recorder.record_metrics(nx_g)

    print("Iteration")
    print(f"0: {nx_g}")

    for i in range(1, ITERATIONS+1):

        # predictions
        to_predict = selector.nodes_to_predict(nx_g)
        predictions = alg.predict(nx_g, directed=DIRECTED, nodes=to_predict)
        add_edges(nx_g, predictions)

        # removals
        to_remove = selector.nodes_to_remove(nx_g)
        removals = rm.removals(nx_g, directed=DIRECTED, nodes=to_remove)
        remove_edges(nx_g, removals)

        # compute metrics
        recorder.record_metrics(nx_g)

        if i % 2 == 0:
            print(f"{i}: {nx_g}")
    
    return recorder


def main():
    # load graph
    init_g = nx.read_edgelist(EDGELIST_PATH, create_using=nx.DiGraph(), nodetype=int)
    if not DIRECTED:
        # keep as DiGraph but add edges in opposite direction
        init_g.add_edges_from([(v, u) for u, v in init_g.edges()])

    # load minorities
    minorities = []
    with open(MINORITIES_PATH, "r") as f:
        for line in f:
            node = int(line.strip())
            minorities.append(node)

    print(f"Minority size: {len(minorities)}")

    # evolve the network
    start = time.time()
    recorder = evolve_network(init_g, minorities)
    end = time.time()
    print(f"Time elapsed: {end - start}")

    # plot metrics
    recorder.plot_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Evolution")
    parser.add_argument("--basename", type=str, default="congress", help="Basename of dataset")
    parser.add_argument("--directed", type=bool, default=True, help="Is the graph directed?")
    args = parser.parse_args()

    BASENAME = args.basename
    DIRECTED = args.directed

    OUTPUT_PREFIX = BASENAME + "." + alg.__name__

    EDGELIST = BASENAME + ".txt"
    MINORITIES = BASENAME + ".minorities"
    EDGELIST_PATH = os.path.join(INPUT_DIR, EDGELIST)
    MINORITIES_PATH = os.path.join(INPUT_DIR, MINORITIES)

    # Seed
    random.seed(42)

    main()

