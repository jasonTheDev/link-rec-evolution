import networkx as nx


ALPHA = 0.85
TOL = 1e-01 # higher tolerance for faster convergence


def initialize(G, directed, protected):
    """
    Initialize the graph for personalized pagerank.
    """
    return G


def predict(G, directed, nodes):
    """
    Predict links using personalized pagerank.
    """
    predictions = []
    tmp_g = G.copy()  # keep track of edges

    # personalized pagerank for each node
    for node in nodes:
        personalization = {node: 1} # start node

        # calculate and sort pagerank scores
        pr_scores = nx.pagerank(G, alpha=ALPHA, personalization=personalization, tol=TOL)
        sorted_scores = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)

        neighbors = set(G.successors(node))

        # find the top edge that doesn't already exist
        for predicted_node, score in sorted_scores:
            if predicted_node != node and predicted_node not in neighbors:
                predictions.append((node, predicted_node))
                tmp_g.add_edge(node, predicted_node)
                if not directed:
                    tmp_g.add_edge(predicted_node, node)
                break
    
    assert len(predictions) == G.number_of_nodes()
    return predictions
