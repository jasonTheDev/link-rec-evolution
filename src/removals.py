import random
import networkx as nx


class RandomEdgeRemover:
    def __init__(self):
        pass
    
    def removals(self, nx_g, directed, nodes):
        """
        Returns a list of random outgoing edges per node
        where both nodes have degree > 1.
        """
        tmp_g = nx_g.copy()
        removals = []

        random.shuffle(nodes)
        for node in nodes:
            if nx_g.out_degree(node) > 1:
                successors = list(tmp_g.successors(node))
                random.shuffle(successors)
                for successor in successors:
                    if tmp_g.in_degree(successor) > 1:
                        removals.append((node, successor))
                        tmp_g.remove_edge(node, successor)
                        if not directed:
                            tmp_g.remove_edge(successor, node)
                        break

        assert len(removals) <= nx_g.number_of_nodes()
        return removals
