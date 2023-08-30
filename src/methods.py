import random
import networkx as nx


# TODO: better name?
class Wagner2022():
    NAME = "wagner" # For Driver    

    def __init__(self, nx_g, directed, protected):
        self.directed = directed
        self._nodes = list(nx_g.nodes())
    
    def nodes_to_predict(self, nx_g):
        """
        Return all nodes.
        """
        return self._nodes.copy()
    
    def edges_to_remove(self, nx_g):
        """
        Random edge per node.
        """
        nodes = self._nodes.copy()
        return select_edges(nx_g, self.directed, nodes)
    

# just an example for now
class OtherMethod():
    NAME = "other" # For Driver
    
    def __init__(self, nx_g, directed, protected):
        self._nodes = list(nx_g.nodes())

    def nodes_to_predict(self, nx_g):
        """
        Return 10% of nodes selected randomly.
        """
        nodes = self._nodes.copy()
        return random.sample(nodes, int(len(nodes) * 0.1))
    
    def edges_to_remove(self, nx_g):
        """
        Return empty list. No removals.
        """
        return []
    

def select_edges(nx_g, directed, nodes):
    """
    Returns a list containing one random outgoing edges per node
    in nodes where in_degree > 1 and out_degree > 1.
    """
    tmp_g = nx_g.copy()
    selected_edges = []

    random.shuffle(nodes)
    for node in nodes:
        if tmp_g.out_degree(node) > 1:
            successors = list(tmp_g.successors(node))
            random.shuffle(successors)
            for successor in successors:
                if tmp_g.in_degree(successor) > 1:
                    selected_edges.append((node, successor))
                    tmp_g.remove_edge(node, successor)
                    if not directed:
                        tmp_g.remove_edge(successor, node)
                    break

    assert len(selected_edges) <= len(nodes)
    return selected_edges
        