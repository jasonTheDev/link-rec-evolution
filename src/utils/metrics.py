import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# eigenvector centrality
# MAX_ITER = 1000
# TOL = 1e-03

# visibility
VISIBILITY_RATIO = 0.1

class Recorder:
    def __init__(self, 
                 directed,
                 protected, 
                 output_dir,
                 output_prefix):
                 
        self.directed = directed
        self.protected = protected

        self.plot_path = os.path.join(output_dir, output_prefix + ".png")
        self.gini_path = os.path.join(output_dir, output_prefix + ".gin")
        self.cluster_path = os.path.join(output_dir, output_prefix + ".clu")
        self.visibility_path = os.path.join(output_dir, output_prefix + ".vis")

    
    def clear_files(self):
        """
        Clears the files for the metrics.
        """
        open(self.gini_path, 'w').close()
        open(self.cluster_path, 'w').close()
        open(self.visibility_path, 'w').close()
        try:
            os.remove(self.plot_path)
        except OSError:
            pass

    
    def record_metrics(self, nx_g):
        """
        Records the gini coefficient, cluster coefficient and protected
        visibility for the given graph.
        """
        gini = gini_of_degree_distribution(nx_g)
        cluster = average_clustering(nx_g)
        visibility = get_visibility(nx_g, self.directed, self.protected)

        append_to_file(self.gini_path, gini)
        append_to_file(self.cluster_path, cluster)
        append_to_file(self.visibility_path, visibility)


    def plot_metrics(self, show=True):
        """
        Plot the gini coefficient, cluster coefficient and minority 
        visibility for each iteration.
        """
        ginis = from_file(self.gini_path)
        clusters = from_file(self.cluster_path)
        visibilities = from_file(self.visibility_path)

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
        
        if show:
            plt.show()

        # write plot to file
        fig.savefig(self.plot_path)
        

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


def get_visibility(nx_g, directed, nodes):
    """
    Returns the visibility of the given nodes.
    """
    if directed:
        ranking_dict = nx.pagerank(nx_g)
    else:
        ranking_dict = nx.pagerank(nx_g)

        # Other option for undirected graphs
        # ranking_dict = nx.eigenvector_centrality(nx_g, max_iter=MAX_ITER, tol=TOL)

    # sort the ranking
    ranking_list = list(ranking_dict.items())
    ranking_list.sort(key=lambda x: x[1], reverse=True)


    # get the top nodes
    num_visible = int(nx_g.number_of_nodes() * VISIBILITY_RATIO)
    visible_nodes = [node for (node, ranking) in ranking_list[:num_visible]]

    # find the fraction of visible nodes
    visibility = len([node for node in nodes if node in visible_nodes]) / float(num_visible)
    return visibility


def average_clustering(nx_g):
    """
    Returns the average clustering coefficient of the graph.
    """
    return nx.average_clustering(nx_g)


def append_to_file(file_path, value):
    with open(file_path, 'a') as f:
        f.write(str(value) + '\n')


def from_file(filename):
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f]
