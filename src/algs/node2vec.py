import networkx as nx
import random
import utils.embedding as embedding


NAME = "node2vec" # For Driver

NUM_WALKS = 5  # Saxena uses 10
WALK_LENGTH = 40  # Saxena uses 80


def initialize(G, directed, protected):
    """
    Nothing to do here.
    """
    return G


def random_walk(nx_g, start_node):
    """
    Simulate random walk starting from a given node.
    """    
    walk = [start_node]
    if len(list(nx_g.successors(start_node))) > 0:
        while len(walk) < WALK_LENGTH:
            cur = walk[-1]
            successors = list(nx_g.successors(cur))
            if successors == []: # directed graph
                break
            next_node, = random.choices(successors)
            walk.append(next_node)
    return walk


def simulate_walks(nx_g):
    """
    Repeatedly run random walks from each node.
    """
    walks = []
    nodes = list(nx_g.nodes())
    for walk_iter in range(NUM_WALKS):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(nx_g, node))
    return walks


def predict(nx_g, directed, nodes, param1=None, param2=None):
    """
    Returns list of predicted edges.
    """
    walks = simulate_walks(nx_g)
    node_vectors = embedding.word2vec(walks)
    predictions = embedding.predict_most_similar(nx_g, node_vectors, directed, nodes)
    return predictions
