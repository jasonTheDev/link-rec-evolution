import networkx as nx
import random
import utils.embedding as embedding

NUM_WALKS = 5  # Saxena uses 10
WALK_LENGTH = 40  # Saxena uses 80
NAME = f"minwalk" # For Driver


def random_walk(G, node):
    """
    classic random walk for cross walk initialization
    """
    walk = [node]
    for _ in range(WALK_LENGTH):
        neighbors = list(G.successors(node))
        if not neighbors:
            break
        node = neighbors[random.choice(range(len(neighbors)))]
        walk.append(node)
    return walk

def setweight(G):
    """
    Set initial weights based on influence scores
    """
    influence_scores = calculate_influence_scores(G)
    for u, v in G.edges():
        G[u][v]['weight'] = (influence_scores[u] + influence_scores[v]) / 2.0
    for u in G.nodes():
        total_outgoing_weight = sum(G[u][v]['weight'] for v in G.neighbors(u))
        for v in G.neighbors(u):
            G[u][v]['weight'] /= float(total_outgoing_weight) if total_outgoing_weight else 1
    return G


def minwalk(G):
    """
    Reweight edges
    """
    G = setweight(G) #initialize
    G = set_valid(G) #initialize

    # Step 1: Calculate closeness to boundary
    m = {}
    for v in G.nodes():
        boundary_count = 0
        for _ in range(NUM_WALKS):
            Wv = random_walk(G, v)
            boundary_count += sum(1 for u in Wv if G.nodes[u]['community']==0)
        boundary_count = boundary_count-NUM_WALKS if G.nodes[v]['community']==0 else boundary_count
        m[v] = boundary_count / float(NUM_WALKS * WALK_LENGTH)
    
    # Step 2: Reweight edges
    for v in G.nodes():
        if G.nodes[v]['valid']==1:
            lv = G.nodes[v]['community']
            Nv = [u for u in G.successors(v) if G.nodes[u]['community'] == lv]
            Rv = [i for i in G.successors(v) if G.nodes[i]['community']!=lv]
            Z = sum(G[v][u].get('weight') * m[u] for u in Nv)

            #Case1: For Edges in same community
            if Z!=0:
                if lv==1:
                    for u in Nv :
                        G[v][u]['weight'] = G[v][u].get('weight') * m[u] / float(Z)
            
            #Case2: Edges connecting different communities
            Z1 = sum(G[v][u].get('weight',1) * m[u] for u in Rv)
            if Z1!=0:
                if lv==1:
                    for u in Rv:
                        G[v][u]['weight'] = G[v][u].get('weight') * m[u] / float(Z1)
    return G



def weighted_random_walk(G, start_node):
    """
    Simulate one weighted random walk
    """
    walk = [start_node]
    current_node = start_node
    
    for _ in range(WALK_LENGTH):
        neighbors = list(G.successors(current_node))
        
        # If the current node has no neighbors, exit
        if not neighbors:
            break
        
        # Choose the next node based on edge weights
        weights = [G[current_node][neighbor].get('weight') for neighbor in neighbors]
        sum_weights = sum(weights)

        # normalize weights
        if sum_weights == 0:
            normalized_weights = [1.0/len(neighbors) for _ in neighbors]
        else:
            normalized_weights = [float(w)/sum_weights for w in weights]

        next_node = random.choices(neighbors, weights=normalized_weights, k=1)[0]
        walk.append(next_node)
        current_node = next_node

    return walk


def walkforeachnode(G):
    """
    Simulate random walks for each node
    """
    cx_G = minwalk(G)
    walks = []
    nodes = list(cx_G.nodes())
    for _ in range(NUM_WALKS):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(weighted_random_walk(cx_G, node))
    return walks

def calculate_influence_scores(G):
    """
    Calculate the influence score for each node based on its neighborhood.
    """
    influence_scores = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            influence_score = sum(G.nodes[neighbor]['community'] == 0 for neighbor in neighbors) / float(len(neighbors))    
        else:
            influence_score = 0
        influence_scores[node] = influence_score
    return influence_scores


def set_valid(G):
    """
    Set valid nodes
    """
    result = get_valid_nodes(G,minority_ratio)
    valid ={node: 1 if node in result\
            else 0 for node in G.nodes()}
    nx.set_node_attributes(G, valid, 'valid')
    return G

def get_valid_nodes(G,ratio_minority_in_all):
    """
    Returns list of valid nodes
    """
    print(ratio_minority_in_all)
    pagerank = nx.pagerank(G)
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    top_10_percent = len(G.nodes()) // 10
    majority_nodes = [node for node in G.nodes() if G.nodes[node]['community'] == 1]
    sorted_majority_nodes = sorted(majority_nodes, key=lambda node: pagerank[node], reverse=True)
    num_top_majority_nodes = int(len(sorted_majority_nodes) * (1-ratio_minority_in_all))
    top_majority_nodes_final = sorted_majority_nodes[:num_top_majority_nodes] 
    top_nodes = sorted_nodes[:top_10_percent]
    valid_majority = [node for node in top_nodes if node not in top_majority_nodes_final \
                    and G.nodes[node]['community']==1]
    
    return valid_majority


def initialize(G,directed, protected):
    """
    set community label for each node
    set global minority _ratio
    """
    global minority_ratio
    minority_ratio =len(protected)/float(G.number_of_nodes())
    minority = set(protected)
    communities = {node: 0 if node in minority else 1 for node in G.nodes()}
    nx.set_node_attributes(G, communities, 'community')

    return G
    
def predict(nx_g, directed, nodes):
    """
    Returns list of predicted edges.
    """
    walks = walkforeachnode(nx_g)
    node_vectors = embedding.word2vec(walks)
    predictions = embedding.predict_most_similar(nx_g, node_vectors, directed, nodes)
    return predictions

