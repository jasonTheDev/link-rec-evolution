from gensim.models import Word2Vec

# Word2Vec
VECTOR_SIZE = 128
WINDOW_SIZE = 5
MIN_COUNT = 0
SG = 1  # 1 for skip-gram
WORKERS = 8


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
