import numpy as np
import networkx as nx
import random

NAME = "ulocal_fair_ppr" # For Driver


class ULFPRfromPaper:
    #initialize
    def __init__(self, graph = None, edge_file = 'out_graph.txt', 
                     community_file = 'out_community.txt',  
                     personalization = 0, alpha = 0.85, 
                     max_iter = 100, undirected = True, epsilon =1e-06):                     
        self.community_file = community_file
        self.personalization = int(personalization)
        self.alpha = alpha
        self.undirected = undirected
        # Step 1 in google colab 
        self.graph = graph if graph is not None \
                     else nx.read_edgelist(edge_file, nodetype=int)\
                     if undirected \
                     else nx.read_edgelist(edge_file, nodetype=int, create_using=nx.DiGraph())
        self.nnode = self.graph.number_of_nodes()
        
        #self.C1,self.C2,self.phi, self.community_dict = self.read_community_phi()
        self.C1,self.C2,self.phi, self.community_dict = self.read_comm_phi_from_graph()
        
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.outC1, self.outC2 = self.compute_out_degrees()
        #self.Lr, self.Lb = self.compute_Lr_Lb() \
        #                    if undirected \
        #                    else self.Di_compute_Lr_Lb()
        self.Lr, self.Lb = self.Di_compute_Lr_Lb()
        self.deltaR,self.deltaB = self.compute_deltas()
        self.PL = self.get_PL()
    
    #Step 1, 2, 3 in google colab
    #Reads the community file and determines which nodes are in community 1 and community 2
    #get phi = ratio of protected group
    def read_community_phi(self):
        nodes, communities = np.loadtxt(self.community_file, dtype=int, unpack=True)
        R = np.where(communities == 0)[0]
        B = np.where(communities == 1)[0]
        phi = len(R) / (len(R) + len(B))
        return set(R),set(B),phi, dict(zip(nodes, communities)) 
    
    def read_comm_phi_from_graph(self):
        nodes = np.array(list(self.graph.nodes()))
        communities = np.array([data['community'] for node, data in self.graph.nodes(data=True)])
        node_community_dict = dict(zip(nodes, communities))
        R = set(nodes[communities == 0])
        B = set(nodes[communities == 1])
        phi = len(R) / len(nodes)
        
        return R,B,phi, node_community_dict 
        
        
    # Step 4 in google colab
    # get out-neighbor of comm1 and comm2 for each node
    def compute_out_degrees(self):
        community = self.community_dict
        outC1 = []
        outC2 = []
        for i in range(self.graph.number_of_nodes()):
            neighbors = set(self.graph.neighbors(i))
            C1_count = len(neighbors & self.C1)
            C2_count = len(neighbors) - C1_count
            outC1.append(C1_count)
            outC2.append(C2_count)
            
        return outC1, outC2
    
    # Step 5 in google colab
    # divide nodes in to two set Lr and Lb
    # if sink node, node belong to Lr and Lb
    # for directed graph
    def Di_compute_Lr_Lb(self):
        outC1, outC2 = self.outC1,self.outC2
        skip = set([node for node in (self.C1 | self.C2) if self.graph.out_degree(node)==0])
        ratioC1 = {i: outC1[i] / (outC1[i]+outC2[i]) for i in range(len(outC1)) if i not in skip}
        Lr=set()
        Lb=set()
        for i in ratioC1.keys():        
            if ratioC1[i]<self.phi:
                Lr.add(i)
            else:
                Lb.add(i)
        Lr |= skip
        Lb |= skip
        return Lr,Lb
    
    # Step 5 in google colab
    # divide nodes in to two set Lr and Lb
    # for undirected graph
    def compute_Lr_Lb(self):
        outC1, outC2 = self.outC1,self.outC2
        ratioC1 = {i: outC1[i] / (outC1[i]+outC2[i]) for i in range(len(outC1))}
        Lr=set()
        Lb=set()
        for i in (self.graph.nodes()):        
            if ratioC1[i]<self.phi:
                Lr.add(i)
            else:
                Lb.add(i)
        return Lr,Lb
    
    # Step 6 in google colab
    # get residual part delatR(protected) and deltaB
    def compute_deltas(self):
        skip = (self.Lr & self.Lb)
        deltaR = [self.phi if i in skip 
                  else (self.phi - (1-self.phi) * self.outC1[i] / self.outC2[i])
                  if i in self.Lr 
                  else 0.0 
                  for i in range(self.nnode)]
                 
        deltaB=[(1-self.phi) if i in skip
                else (1-self.phi) - self.phi * self.outC2[i] / self.outC1[i]
                if i in self.Lb
                else 0.0
                for i in range(self.nnode)]
        return deltaR, deltaB
    
    # Step 8 in google colab
    # get original personalization pagerank from networkx
    # used for calculating residual policy vector x and y
    # default alpha = 0.85, max_iter=100, eps = 1e-06
    def get_initial_pr(self):
        p = {self.personalization:0.15}
        return nx.pagerank(self.graph, alpha=self.alpha, personalization=p)
    
    # Step 9 in google colab
    # get residual policy vector x and y
    # use x and y to get X = outer(deltaR, x) and Y = outer(deltaB,y)
    def get_x_y(self):
        PO = self.get_initial_pr()
        
        
        x = [1/len(self.C1) if i in self.C1
            else 0.0
            for i in range(self.nnode)] 
        
        y = [1/len(self.C2) if i in self.C2 
            else 0.0
            for i in range(self.nnode)]
        X = np.outer(self.deltaR, x)
        Y = np.outer(self.deltaB, y)  
        return X,Y
    
    
    # Step 7 in google colab
    # get transition matrix PL
    def get_PL(self):
        N = self.graph.number_of_nodes()

        # Initialize a transition probability matrix full of zeros
        P_L = np.zeros((N, N))

        # For each edge in the graph
        for i, j in self.graph.edges():
            # If node i is in set LR
            if i in self.Lr:
                P_L[i][j] = (1 - self.phi) / self.outC2[i]

            # If node i is in set LB
            else:
                P_L[i][j] = self.phi / self.outC1[i]
        return P_L
    
    # delete edges from networkx graph
    # work for directed graph
    # makes sure the degree at the end for each node is NOT 0
    def Di_delete(self):
        count=0
        delete = []      
        skip = [node for node in self.graph.nodes() if self.graph.out_degree(node) <= 1]
        initial = len(skip)
        skipped = []
        skip_neigh = [node for node in skip if self.graph.in_degree(node)<=1]
        for node in range(len(self.graph.nodes())):
            if node not in skip:
                neighbors = [neighbor for neighbor in self.graph.neighbors(node) if neighbor not in skip_neigh]
                if neighbors:
                    chosen_neighbor = random.choice(neighbors)
                    self.graph.remove_edge(node, chosen_neighbor)
                    delete.append(node)
                    if self.graph.in_degree(chosen_neighbor) <= 1 and chosen_neighbor in skip:
                        skip_neigh.append(chosen_neighbor)
                else:
                    skipped.append(node)
            else:
                skipped.append(node)
        for n in self.graph.nodes():
            if self.graph.degree(n)==0:
                count+=1
       
        with open("output/tracking.txt", "a") as f:
            f.write("\ndeletion starts: \n")
            f.write("nodes with no more than 1 neighbor initially: "+str(initial)+"\n")
            f.write("deleting: "+str(len(delete))+" edges\n")
            f.write("skip "+str(len(skipped))+" nodes: \n")
            f.write(" ".join(map(str, skipped)))
            f.write("\nafter deletion, number of nodes with 0 link: "+str(count)+"\n")
    
    
    # delete edges from networkx graph
    # work for undirected graph
    # makes sure the degree at the end for each node is NOT 0
    def delete(self): 
        count=0
        delete = []      
        skip = [node for node in self.graph.nodes() if self.graph.degree(node) <= 1]
        initial = len(skip)
        skipped =  []       
        for node in range(len(self.graph.nodes())):
            if node not in skip:
                neighbors = [neighbor for neighbor in self.graph.neighbors(node) \
                            if self.graph.degree(neighbor) > 1]
                                
                if neighbors:  # If there are eligible neighbors
                    chosen_neighbor = random.choice(neighbors)
                    self.graph.remove_edge(node, chosen_neighbor)
      
                    delete.append(node)
                    if self.graph.degree(chosen_neighbor) <= 1:
               
                        skip.append(chosen_neighbor)
                else:
                    
                    skipped.append(node)
            else:
                skipped.append(node)
    
       
        for n in self.graph.nodes():
            if self.graph.degree(n)==0:
                count+=1
       
        with open("output/tracking.txt", "a") as f:
            f.write("\ndeletion starts: \n")
            f.write("nodes with no more than 1 neighbor initially: "+str(initial)+"\n")
            f.write("deleting: "+str(len(delete))+" edges\n")
            f.write("skip "+str(len(skipped))+" nodes: \n")
            f.write(" ".join(map(str, skipped)))
            f.write("\nafter deletion, number of nodes with 0 link: "+str(count)+"\n")
    
    # addedge to networkx graph (doesn't change original file)
    # work for both undirected graph and directed graph
    def addedge(self, elist,filename):
        self.graph.add_edges_from(elist)
        with open("output/tracking.txt", "a") as f:
            f.write("\nadding starts: \n")
            f.write("adding: "+str(len(elist))+" edges\n\n")
            f.write("after add and del current edges: "+str(self.graph.number_of_edges())+"\n")
            f.write("after add and del current nodes: "+str(self.graph.number_of_nodes())+"\n")
            f.write("\n\n\n")
        with open(filename, "w") as f1:
            for t in elist:
                f1.write(f"{t[0]} {t[1]}\n")
    
    # write updated graph into output/graph/graph{i}.txt
    def write_edges_to_file(self, filename):
        with open(filename, 'w') as f:
            for edge in self.graph.edges():
                f.write(f"{edge[0]} {edge[1]}\n")
                
                
    # work for both undirected and directed graph
    # call all functions and return prediction SET (not order)
    def process(self,nodes_to_predict=None):
        if not nodes_to_predict: 
            iterable = range(0, self.nnode)
        else:
            iterable = sorted(nodes_to_predict)
        # 7 steps done when initialing:
        # Step 1: Read files and preprocess data
        # Step 2: Identify Red and Blue groups
        # Step 3: Compute phi
        # Step 4: Calculate out-degrees for each group
        # Step 5: Compute Lr and Lb groups
        # Step 6: Calculate residual delta_R and delta_B
        # Step 7: Calculate matrix P_L
        
        
        # Step 8: Get Initial Pagerank set as 1/nnode for all nodes (Step 10 in google colab)
        pr_init = np.full((1,self.nnode), 1.0/self.nnode)
        
        prediction = set()
        #for a in range(0,2):
        for a in iterable:
            print("now node: "+str(a)) if a%50==0 else None
            self.personalization = a
            
            # Step 9: Calculate x and y
            # Step 9.1: Compute matrices X_P = outer(deltaR,x) and Y_P = outer(deltaB,y)
            # Step 10: Get Original node personalization pagerank from networkx (step 8 in google colab)
            X_P,Y_P = self.get_x_y()

            # Step 11: Initialize personalization vector v_T
            
            v = np.zeros(self.nnode)
            v[self.personalization] = 1.0
            v = self.alpha*(v)
            finalmatrix = self.PL+X_P+Y_P
            
            # Step 12: iteration function
            iteration = 0
            gamma = (1-self.alpha)
            while(iteration<100):
                pr = gamma*(np.dot(pr_init,finalmatrix))+v
                if sum(abs(pr.flatten()-pr_init.flatten()))<self.epsilon:
                    break
                iteration+=1
                pr_init = pr.copy()
            # Step 14: get recommended one link for current node
            pr = pr.T
            '''
            with open('lfprfrompaper.txt', "w") as f1:
                for index, value in enumerate(pr):
                    f1.write(f"Index: {index}, Value: {value[0]}\n")
            '''
            max_key = np.argmax(pr)
            
            
            #pr = pr / np.sum(pr) #normalized function
            
            # get prediction set
            if self.undirected:
                while self.graph.has_edge(a, max_key) or a==max_key \
                    or ((a,max_key) in prediction) or((max_key,a) in prediction):

                    pr = np.delete(pr, max_key, axis=0)
                    max_key = np.argmax(pr)
                prediction.add((a,max_key))
            else:
                while self.graph.has_edge(a, max_key) or a==max_key \
                    or ((a,max_key) in prediction):

                    pr = np.delete(pr, max_key, axis=0)
                    max_key = np.argmax(pr)
                prediction.add((a,max_key))
        return prediction
        

       
# read community 
# G will be a graph
# directed boolean
# minority a list of ids
def initialize(G, directed, protected):
    minority = set(protected)
    communities = {node: 0 if node in minority else 1 for node in G.nodes()}
    nx.set_node_attributes(G, communities, 'community')
    return G
    
    
def predict(G, directed,nodes):
    ulfpr = ULFPRfromPaper(graph = G, undirected = not directed)
    return list(ulfpr.process(nodes))