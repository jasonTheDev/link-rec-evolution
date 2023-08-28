import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


VISIBILITY_RATIO = 0.1


def gini_of_list(x):
    """
    Returns the Gini coefficient of the elements of a numpy array.
    """
    # use int64 to avoid overflow
    x = x.astype(np.float64)

    total = np.int64(0)
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))


def gini_of_degree_distribution(nx_g):
    """
    Returns the Gini coefficient of the in degree distribution.
    """
    in_degrees = np.array([nx_g.in_degree(node) for node in nx_g.nodes()])
    return gini_of_list(in_degrees)


def pagerank_visibility(nx_g, nodes):
    """
    Returns the fraction of nodes in the top VISIBILITY_RATIO of pagerank.
    """
    ranking_dict = nx.pagerank(nx_g)
    return get_visibility(nx_g, nodes, ranking_dict)


def eigenvector_visibility(nx_g, nodes):
    """
    Returns the fraction of nodes in the top VISIBILITY_RATIO of eigenvector centrality.
    """
    ranking_dict = nx.eigenvector_centrality(nx_g)
    return get_visibility(nx_g, nodes, ranking_dict)


def get_visibility(nx_g, nodes, ranking_dict):
    """
    Returns the visibility of the given nodes.
    """
    # sort the ranking
    ranking_list = list(ranking_dict.items())
    ranking_list.sort(key=lambda x: x[1], reverse=True)

    # get the top nodes
    num_visible = int(nx_g.number_of_nodes() * VISIBILITY_RATIO)
    visible_nodes = [node for (node, ranking) in ranking_list[:num_visible]]

    # find the fraction of visible nodes
    visibility = float(len([node for node in nodes if node in visible_nodes]) / num_visible)
    return visibility


def average_clustering(nx_g):
    """
    Returns the average clustering coefficient of the graph.
    """
    return nx.average_clustering(nx_g)


def plot_to_file(file_path, ginis, clusters, visibilities):
    """
    Plot the gini coefficient, cluster coefficient and minority 
    visibility for each iteration.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(ginis)
    ax[0].set_title("Gini Coefficient")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Gini Coefficient")
    ax[0].set_ylim(0, 1)
    ax[1].plot(clusters)
    ax[1].set_title("Cluster Coefficient")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Cluster Coefficient")
    ax[1].set_ylim(0, 1)
    ax[2].plot(visibilities)
    ax[2].set_title("Minority Visibility")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Fraction of Minority Nodes")
    ax[2].set_ylim(0, 1)
    plt.show()

    # write plot to file
    fig.savefig(file_path)


def list_to_file(file_path, values):
    with open(file_path, 'w') as f:
        for value in values:
            f.write(str(value) + '\n')
