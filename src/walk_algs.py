from gensim.models import Word2Vec
import networkx as nx
import random

# local imports
import communities as cm

# Word2Vec
VECTOR_SIZE = 128
WINDOW_SIZE = 5
MIN_COUNT = 0
SG = 1  # 1 for skip-gram
WORKERS = 8

NUM_WALKS = 5  # Saxena uses 10
WALK_LENGTH = 40  # Saxena uses 80

ALPHA = 1.0
BETA = 2.0


class Node2Vec():
    def __init__(self, nx_g, directed, protected):
        self.directed = directed

    def predict(self, nx_g, directed, nodes):
        """
        Returns list of predicted edges.
        """
        walks = simulate_node2vec_walks(nx_g)
        node_vectors = word2vec(walks)
        predictions = predict_most_similar(nx_g, node_vectors, directed, nodes)
        return predictions
    

class NodeSim():
    def __init__(self, nx_g, directed, protected):
        self.directed = directed

        if directed:
            comms = cm.directed_comms(G)
        else:
            comms = cm.undirected_comms(G)
        G = cm.add_community_labels(G, comms)

    def predict(self, nx_g, directed, nodes):
        """
        Returns list of predicted edges.
        """
        walks = simulate_nodesim_walks(nx_g)
        node_vectors = word2vec(walks)
        predictions = predict_most_similar(nx_g, node_vectors, directed, nodes)
        return predictions
    

class FairWalk():
    def __init__(self, nx_g, directed, protected):
        self.directed = directed
        self.protected = protected.copy()

        is_protected = {}
        for node in protected:
            is_protected[node] = True

        for node in nx_g.nodes():
            if node not in is_protected:
                is_protected[node] = False

        self.is_protected = is_protected
    
    def predict(self, nx_g, directed, nodes):
        """
        Returns list of predicted edges.
        """
        walks = simulate_fair_walks(nx_g, self.is_protected)
        node_vectors = word2vec(walks)
        predictions = predict_most_similar(nx_g, node_vectors, directed, nodes)
        return predictions
    

def simulate_fair_walks(nx_g, is_protected):
    """
    Repeatedly run random walks from each node.
    """
    walks = []
    nodes = list(nx_g.nodes())
    for walk_iter in range(NUM_WALKS):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(fair_walk(nx_g, node, is_protected))
    return walks


def fair_walk(nx_g, start_node, is_protected):
    """
    Give equal chance to protected nodes at each step.
    """
    walk = [start_node]
    while len(walk) < WALK_LENGTH:
        cur = walk[-1]
        successors = list(nx_g.successors(cur))
        if successors == []: # directed graph
            break

        protected_successors = []
        unprotected_successors = []
        for successor in successors:
            if is_protected[successor]:
                protected_successors.append(successor)
            else:
                unprotected_successors.append(successor)
                
        if protected_successors == [] or unprotected_successors == []:
            next_node, = random.choices(successors)
        else:
            # 50% chance of choosing protected node
            if random.random() < 0.5:
                next_node, = random.choices(protected_successors)
            else:
                next_node, = random.choices(unprotected_successors)
        walk.append(next_node)
    return walk
    

def random_walk(nx_g, start_node):
    """
    Simulate random walk starting from a given node.
    """    
    walk = [start_node]
    while len(walk) < WALK_LENGTH:
        cur = walk[-1]
        successors = list(nx_g.successors(cur))
        if successors == []: # directed graph
            break
        next_node, = random.choices(successors)
        walk.append(next_node)
    return walk


def simulate_node2vec_walks(nx_g):
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


def nodesim_walk(G, start_node, nghs, probs):
    """
    Simulate nodesim random walk starting from a given node.
    """
    walk = [start_node]
    while len(walk) < WALK_LENGTH:
        cur = walk[-1]
        if nghs[cur] == []: # directed graphs
            break
        next_node, = random.choices(nghs[cur], probs[cur])
        walk.append(next_node)
    return walk


def simulate_nodesim_walks(G):
    """
    Repeatedly run random walks from each node.
    """
    nghs, probs = compute_nodesim_edge_probs(G)

    walks = []
    nodes = list(G.nodes())
    for walk_iter in range(NUM_WALKS):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(nodesim_walk(G, node, nghs, probs))
    return walks


# ADAPTED: faster computation of edge probabilities
def compute_nodesim_edge_probs(G):
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


def word2vec(walks):
    """
    Learn network embeddings using Skipgram model.
    """
    # Run the following line if the node type is not string
    new_walks = [[str(i) for i in walk] for walk in walks]

    # Parameters adapted for notebook
    model = Word2Vec(new_walks,
                    vector_size=VECTOR_SIZE,
                    window=WINDOW_SIZE,
                    min_count=MIN_COUNT,
                    sg=SG,
                    workers=WORKERS,)

    return model.wv


def predict_most_similar(nx_g, node_vectors, directed, nodes):
    """
    Returns list of predicted edges.
    """
    tmp_g = nx_g.copy()
    predictions = []
    for node in nodes:
        # out_degree + 1 guarantees that at least one edge will be added
        max_pred = tmp_g.out_degree(node) + 1
        most_similar_nodes = node_vectors.most_similar(str(node), topn=max_pred)

        # add the top edge that doesn't already exist
        for similar_node, similarity in most_similar_nodes:
            similar_node = int(similar_node)
            if not tmp_g.has_edge(node, similar_node):
                predictions.append((node, similar_node))
                tmp_g.add_edge(node, similar_node)
                if not directed:
                    # add opposite edge for undirected case
                    tmp_g.add_edge(similar_node, node)
                break

    assert len(predictions) == len(nodes)   
    return predictions


