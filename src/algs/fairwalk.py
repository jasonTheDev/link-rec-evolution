import networkx as nx
import random
import utils.embedding as embedding

NAME = "fairwalk" # For Driver

NUM_WALKS = 5
WALK_LENGTH = 40

def initialize(G, directed, protected):
    """
    Add a protected attribute to each node
    """
    protected_dict = {}
    for node in G.nodes():
        assert type(node) == int
        protected_dict[node] = False
    for node in protected:
        assert type(node) == int
        protected_dict[node] = True

    nx.set_node_attributes(G, protected_dict, 'protected')
    return G


def fair_walk(nx_g, start_node):
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
            
            # choose from either group with 50% probability
            chosen_group = []
            if random.random() < 0.5:
                # protected chosen
                for successor in successors:
                    if nx_g.nodes[successor]['protected']:
                        chosen_group.append(successor)
            else:
                # unprotected chosen
                for successor in successors:
                    if not nx_g.nodes[successor]['protected']:
                        chosen_group.append(successor)

            if chosen_group == []:
                next_node = random.choice(successors)
            else:
                next_node = random.choice(chosen_group)
            
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
            walks.append(fair_walk(nx_g, node))
    return walks


def predict(nx_g, directed, nodes):
    """
    Returns list of predicted edges.
    """
    walks = simulate_walks(nx_g)
    node_vectors = embedding.word2vec(walks)
    predictions = embedding.predict_most_similar(nx_g, node_vectors, directed, nodes)
    return predictions
