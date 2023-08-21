import networkx as nx


ALPHA = 0.85
MAX_ITER = 1000 # raises error if reached
TOL = 1e-01


def initialize(G, directed):
    """
    Initialize the graph for personalized pagerank.
    """
    G


def predict(G, directed):
    """
    Predict links using personalized pagerank.
    """
    predictions = []
    tmp_g = G.copy() # keep track of edges

    # run personalized pagerank for each node
    for node in G.nodes():
        personalization = {node: 1}
        pr_scores = nx.pagerank(G, alpha=ALPHA, personalization=personalization, max_iter=MAX_ITER, tol=TOL)

        # sort nodes by pagerank score
        sorted_nodes = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
        
        for predicted_node, score in sorted_nodes:
            predicted_node = int(predicted_node)
            if not tmp_g.has_edge(node, predicted_node):
                predictions.append((node, predicted_node))
                tmp_g.add_edge(node, predicted_node)
                if not directed:
                    # add opposite edge for undirected case
                    tmp_g.add_edge(predicted_node, node)
                break
    
    assert len(predictions) == G.number_of_nodes()
    return predictions
