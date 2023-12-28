import networkx as nx
import random
import utils.embedding as embedding

NUM_WALKS = 5  # Saxena uses 10
WALK_LENGTH = 40  # Saxena uses 80
ALPHA = 0.5
P = 2

NAME = "crosswalk" # For Driver

# classic random walk for cross walk initialization
def random_walk(G, node):
    walk = [node]
    for _ in range(WALK_LENGTH):
        neighbors = list(G.successors(node))
        if not neighbors:
            break
        node = neighbors[random.choice(range(len(neighbors)))]
        walk.append(node)
    return walk

def setweight(G):
    influence_scores = calculate_influence_scores(G)
    for u, v in G.edges():
        G[u][v]['weight'] = (influence_scores[u] + influence_scores[v]) / 2
    for u in G.nodes():
        total_outgoing_weight = sum(G[u][v]['weight'] for v in G.neighbors(u))
        for v in G.neighbors(u):
            G[u][v]['weight'] /= total_outgoing_weight if total_outgoing_weight else 1
    return G

# Use CrossWalk algo to resize weight
def crosswalk(G):
    G = setweight(G)
    # Step 1: Calculate closeness to boundary
    m = {}
    for v in G.nodes():
        boundary_count = 0
        for _ in range(NUM_WALKS):
            Wv = random_walk(G, v)
            boundary_count += sum(1 for u in Wv if G.nodes[v]['community'] != G.nodes[u]['community'])
        m[v] = boundary_count / (NUM_WALKS * WALK_LENGTH)
    # Step 2: Reweight edges
    for v in G.nodes():
        lv = G.nodes[v]['community']
        Nv = [u for u in G.successors(v) if G.nodes[u]['community'] == lv]
        Rv = set([G.nodes[u]['community'] for u in G.successors(v) if G.nodes[u]['community'] != lv])
        Z = sum(G[v][u].get('weight') * m[u]**P for u in Nv)
        #Case1: For Edges in same community
        if Z!=0:
            for u in Nv :
                G[v][u]['weight'] = G[v][u].get('weight') * (1 - ALPHA) * m[u]**P / Z
        #Case2: Edges connecting different communities
        for c in Rv:
            Nc_v = [u for u in G.successors(v) if G.nodes[u]['community'] == c and G.nodes[v]['community'] != c]
            Z1 = len(Rv) * sum(G[v][u].get('weight') * m[u]**P for u in Nc_v)
            if Z1!=0:
                for u in Nc_v:
                    if Nv:
                        G[v][u]['weight'] = G[v][u].get('weight') * ALPHA * m[u]**P / Z1
                    else:
                        G[v][u]['weight'] = G[v][u].get('weight') * m[u]**P / Z1 
    return G



def weighted_random_walk(G, start_node):
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
        #normalized_weights = [w/sum_weights for w in weights]

        # if sum_weights=0 set it to be 1/len(neighbors) for each node.
        if sum_weights == 0:
            normalized_weights = [1/len(neighbors) for _ in neighbors]
        else:
            normalized_weights = [w/sum_weights for w in weights]
        next_node = random.choices(neighbors, weights=normalized_weights, k=1)[0]
        walk.append(next_node)
        current_node = next_node

    return walk

# simulate random walks for each node
def walkforeachnode(G):
    cx_G = crosswalk(G)
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
            influence_score = sum(G.nodes[neighbor]['community'] == 0 for neighbor in neighbors) / len(neighbors)
        else:
            influence_score = 0
        influence_scores[node] = influence_score
    return influence_scores

    

# set community label for each node
# set edge weight to be 1 for each edge
def initialize(G, directed, protected):
    minority = set(protected)
    communities = {node: 0 if node in minority else 1 for node in G.nodes()}
    nx.set_node_attributes(G, communities, 'community')
    return G
    
def predict(nx_g, directed, nodes, param1=ALPHA, param2=P):
    """
    Returns list of predicted edges.
    """
    ALPHA = param1
    P = param2
    walks = walkforeachnode(nx_g)
    node_vectors = embedding.word2vec(walks)
    predictions = embedding.predict_most_similar(nx_g, node_vectors, directed, nodes)
    return predictions

