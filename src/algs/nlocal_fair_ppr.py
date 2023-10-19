import numpy as np
import networkx as nx
import random

NAME = "nlocal_fair_ppr" # For Driver


class NLFPRfromPaper:
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
    def Pr_Pb(self):
        Pr = np.zeros((self.nnode, self.nnode))
        edgeset = set(self.graph.edges())
        for i in range(self.nnode):
            for j in range(self.nnode):
                if(i,j) in edgeset and j in self.C1:
                    Pr[i][j] = 1 / self.outC1[i]
                elif self.outC1[i]==0 and j in self.C1:
                    Pr[i][j] = 1/len(self.C1)
                else:
                    Pr[i][j] = 0
        Pb = np.zeros((self.nnode, self.nnode))    

        for i in range(self.nnode):
            for j in range(self.nnode):
                if(i,j) in edgeset and j in self.C2:
                    Pb[i][j] = 1 / self.outC2[i]
                elif self.outC2[i]==0 and j in self.C2:
                    Pb[i][j] = 1/len(self.C2)
                else:
                    Pb[i][j] = 0
        return Pr, Pb
    # Step 7 in google colab
    # get transition matrix PL
    def get_PL(self):
        Pr, Pb = self.Pr_Pb()
        P_L = self.phi*Pr+(1-self.phi)*Pb
        return P_L
                  
                
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
            print("now node: "+str(a)) if a%500==0 else None
            self.personalization = a
            
            # Step 9: Calculate x and y
            # Step 9.1: Compute matrices X_P = outer(deltaR,x) and Y_P = outer(deltaB,y)
            # Step 10: Get Original node personalization pagerank from networkx (step 8 in google colab)

            # Step 11: Initialize personalization vector v_T
            
            
            v = np.zeros(self.nnode)
            for i in self.C1:
                v[i]  = self.phi/len(self.C1)
            for i in self.C2:
                v[i]  = (1-self.phi)/len(self.C2)
            v = self.alpha*(v)
            # Step 12: iteration function
            iteration = 0
            gamma = (1-self.alpha)
            while(iteration<100):
                pr = gamma*(np.dot(pr_init,self.PL))+v
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
    nlfpr = NLFPRfromPaper(graph = G, undirected = not directed)
    return list(nlfpr.process(nodes))