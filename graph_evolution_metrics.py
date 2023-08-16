import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


PAGERANK_RATIO = 0.1


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
    Returns the fraction of nodes in the top PAGERANK_RATIO of pagerank.
    """
    # get number of nodes in top percent
    num_nodes_in_top = int(nx_g.number_of_nodes() * PAGERANK_RATIO)
    # run pagerank
    pr = nx.pagerank(nx_g)
    # convert pagerank to list and sort
    pr_list = list(pr.items())
    pr_list.sort(key=lambda x: x[1], reverse=True)
    # get the top nodes
    top_pr_list = [node for (node, ranking) in pr_list[:num_nodes_in_top]]
    # find the fraction of given nodes in top pagerank
    visibility = float(len([node for node in nodes if node in top_pr_list]) / num_nodes_in_top)
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
