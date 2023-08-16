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


def predict_most_similar(nx_g, node_vectors):
    """
    Returns list of predicted edges.
    """
    predictions = []
    for node in nx_g.nodes():
        # out_degree + 1 guarantees that at least one edge will be added
        max_pred = nx_g.out_degree(node) + 1
        most_similar_nodes = node_vectors.most_similar(str(node), topn=max_pred)

        # add the top edge that doesn't already exist
        for similar_node, similarity in most_similar_nodes:
            if not nx_g.has_edge(node, int(similar_node)):
                predictions.append((node, int(similar_node)))
                break

    # print(f"Predictions {len(predictions)} / {nx_g.number_of_nodes()} nodes")
    assert len(predictions) == nx_g.number_of_nodes()

        
    return predictions