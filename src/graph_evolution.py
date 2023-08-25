import argparse
import os
import random
import time
import numpy as np
import networkx as nx

# local imports
import graph_evolution_metrics as metrics

## CHANGE TO MATCH ALG AND DATASET
import node2vec as alg # addition alg
import random_removal_per_node as rm # removal alg

# Graph Evolution
ITERATIONS = 30

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
    # initial metrics
    ginis = [metrics.gini_of_degree_distribution(nx_g)]
    clusters = [nx.average_clustering(nx_g)]
    visibilities = [metrics.pagerank_visibility(nx_g, minorities)]

    # prepare for algorithm
    nx_g = alg.initialize(nx_g, directed=DIRECTED)

    print("Iteration")
    print(f"0: {nx_g}")

    for i in range(1, ITERATIONS+1):

        predictions = alg.predict(nx_g, directed=DIRECTED)
        add_edges(nx_g, predictions)

        removals = rm.removals(nx_g, directed=DIRECTED)
        remove_edges(nx_g, removals)

        # compute metrics
        ginis.append(metrics.gini_of_degree_distribution(nx_g))
        clusters.append(metrics.average_clustering(nx_g))
        visibilities.append(metrics.pagerank_visibility(nx_g, minorities))
  
        if i % 2 == 0:
            print(f"{i}: {nx_g}")
    
    return ginis, clusters, visibilities


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
    ginis, clusters, visibilities = evolve_network(init_g, minorities)
    end = time.time()
    print(f"Time elapsed: {end - start}")

    # plot data
    metrics.plot_to_file(PLOT_PATH, ginis, clusters, visibilities)
    metrics.list_to_file(GINI_PATH, ginis)
    metrics.list_to_file(CLUSTERING_PATH, clusters)
    metrics.list_to_file(VISIBILITY_PATH, visibilities)

    with open(SUMMARY_PATH, 'w') as f:
        f.write(f"Iterations: {ITERATIONS}\n")
        f.write(f"Removals: {rm.__name__}\n")
        f.write(f"Predictions: {alg.__name__}\n")
        f.write(f"Total number of nodes: {init_g.number_of_nodes()}\n")
        f.write(f"Minority size: {len(minorities)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Evolution")
    parser.add_argument("--basename", type=str, default="congress", help="Basename of dataset")
    parser.add_argument("--directed", type=bool, default=True, help="Is the graph directed?")
    args = parser.parse_args()

    BASENAME = args.basename
    DIRECTED = args.directed

    OUTPUT_PREFIX = BASENAME + "." + alg.__name__
    
    CLUSTERING_PATH = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + ".clu")
    GINI_PATH = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + ".gin")
    VISIBILITY_PATH = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + ".vis")
    SUMMARY_PATH = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + ".sum")
    PLOT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX + ".png")

    EDGELIST = BASENAME + ".txt"
    MINORITIES = BASENAME + ".minorities"
    EDGELIST_PATH = os.path.join(INPUT_DIR, EDGELIST)
    MINORITIES_PATH = os.path.join(INPUT_DIR, MINORITIES)

    # Seed
    random.seed(42)

    main()

