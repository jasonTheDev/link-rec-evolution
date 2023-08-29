# Adapted from: https://github.com/akratiiet/NodeSim

import embedding
import random
import networkx as nx

# local imports
import communities as cm

ALPHA = 1.0
BETA = 2.0

NUM_WALKS = 5  # Saxena uses 10
WALK_LENGTH = 40  # Saxena uses 80


def initialize(G, directed, protected):
    """
    Initialize the graph for nodesim random walks.
    """
    if directed:
        comms = cm.directed_comms(G)
    else:
        comms = cm.undirected_comms(G)
    G = cm.add_community_labels(G, comms)
    return G


# ADAPTED: faster computation of edge probabilities
def compute_edge_probs(G):
    """
    Compute transition probabilities for nodesim random walks.
    """
    comm_att = 'community'

    # ADAPTED: changed from list to dictionary (x20 faster)
    def compute_jc(s1, s2):
        union_size = len(s1 | s2)
        if union_size == 0:
            return 0
        return len(s1 & s2) / union_size

    # Initialize dictionaries
    nghs = {}
    probs = {}
    nghs_set = {}  # Faster Jaccard computation

    # Populate dictionaries
    for node in G.nodes():
        nghs[node] = []
        probs[node] = []
        nghs_set[node] = set(G.successors(node))

    for node in G.nodes():
        for ngh in G.successors(node):
            jaccard = compute_jc(nghs_set[node], nghs_set[ngh])
            pval = jaccard + 1.0 / G.out_degree(node)

            # Check community attributes exist and are equal
            if (comm_att in G.nodes[node] and
                comm_att in G.nodes[ngh] and
                G.nodes[node][comm_att] == G.nodes[ngh][comm_att]):
                weight = ALPHA
            else:
                weight = BETA

            weighted_prob = weight * pval

            # Add to the dictionaries
            nghs[node].append(ngh)
            probs[node].append(weighted_prob)
        
        total = sum(probs[node])
        probs[node] = [x / total for x in probs[node]]

    # Sanity checks
    for node in G.nodes():
        assert len(probs[node]) == len(nghs[node]) == len(nghs_set[node])

    return nghs, probs


def nodesim_walk(G, start_node, nghs, probs):
    """
    Simulate nodesim random walk starting from a given node.
    """
    walk = [start_node]
    # CHANGED: end walk if start node has no neighbors
    # this causes an error in the original code
    if len(nghs[start_node]) > 0:
        while len(walk) < WALK_LENGTH:
            cur = walk[-1]
            if nghs[cur] == []: # directed graphs
                break
            next_node, = random.choices(nghs[cur], probs[cur])
            walk.append(next_node)
    return walk


def simulate_walks(G):
    """
    Repeatedly run random walks from each node.
    """
    nghs, probs = compute_edge_probs(G)

    walks = []
    nodes = list(G.nodes())
    for walk_iter in range(NUM_WALKS):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(nodesim_walk(G, node, nghs, probs))
    return walks


def predict(nx_g, directed, nodes):
    """
    Returns list of predicted edges.
    """
    walks = simulate_walks(nx_g)
    node_vectors = embedding.word2vec(walks)
    predictions = embedding.predict_most_similar(nx_g, node_vectors, directed, nodes)
    return predictions
