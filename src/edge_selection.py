import random
import networkx as nx


class DegreesGreaterThanOne():
    def __init__(self, init_g, directed, protected):
        pass
    
    def select_edges(self, nx_g, directed, nodes):
        """
        Returns a list of random outgoing edges per node
        in nodes where both have degree > 1.
        """
        tmp_g = nx_g.copy()
        selected_edges = []

        random.shuffle(nodes)
        for node in nodes:
            if nx_g.out_degree(node) > 1:
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
