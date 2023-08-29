import networkx as nx

class SelectAll():
    def __init__(self, nx_g, directed, protected):
        """
        Initialize.
        """
        self.nodes = list(nx_g.nodes())
    
    def nodes_to_predict(self, nx_g):
        """
        Return all nodes.
        """
        return self.nodes.copy()
    
    def nodes_to_remove(self, nx_g):
        """
        Return empty list.
        """
        return self.nodes.copy()