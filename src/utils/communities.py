import leidenalg
import networkx as nx
import networkx.algorithms.community as nx_comm
from igraph import Graph


def add_community_labels(G, comms):
    """
    Adds community labels to the NetworkX graph.
    """
    node_to_comm = {}

    for comm_index, comm_nodes in enumerate(comms):
        for node in comm_nodes:
            node_to_comm[node] = comm_index
    nx.set_node_attributes(G, node_to_comm, "community")
    return G


def undirected_comms(G):
    """
    Returns a list of communities in the graph, as detected by Louvain.
    """
    return nx_comm.louvain_communities(G)


def directed_comms(G):
    """
    Returns a list of communities in the graph, as detected by leidenalg.
    """
    # convert to igraph
    G_igraph = Graph.TupleList(G.edges(), directed=True)
    part = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
    # dict
    comms_dict = {}
    for i, comm in enumerate(part.membership):
        if comm not in comms_dict:
            comms_dict[comm] = []
        comms_dict[comm].append(i)
    # list
    comms = []
    for comm in comms_dict.values():
        comms.append(comm)
    return comms