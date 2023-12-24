import argparse
import email
import os
from pprint import pprint
import time
import networkx as nx
import math

# local imports
from utils.metrics import Recorder


# methods
WAGNER = "from methods import Wagner2022 as Method"
OTHERMETHOD = "from methods import OtherMethod as Method" # just an example for now

# datasets
CONGRESS = "congress", True
EMAIL_EU = "email_eu", True
WIKI_VOTE = "wiki_vote", True
FACEBOOK = "facebook", False
LASTFM = "lastfm", False
DEEZER = "deezer", False

# walk based algorithms
NODE2VEC = "import algs.node2vec as alg"
NODESIM = "import algs.nodesim as alg"
FAIRWALK = "import algs.fairwalk as alg"

CROSSWALK = "import algs.crosswalk as alg"
NEWCW = "import algs.newcw as alg"
FIXEDNEWCW1 = "import algs.fixednewcw1 as alg"
FIXEDNEWCW2 = "import algs.fixednewcw2 as alg"
MINWALK = "import algs.mergecw as alg"
MINWALKNOP = "import algs.MinWalknop as alg"
TESTWALK = "import algs.TestWalk as alg"
SIMPLEWALK = "import algs.minwalk as alg"
# pagerank based algorithms
PPR = "import algs.ppr as alg"
ULOCALPPR = "import algs.ulocal_fair_ppr as alg"
NLOCALPPR = "import algs.nlocal_fair_ppr as alg"
PLOCALPPR = "import algs.plocal_fair_ppr as alg"

PERCENT05 = "05percent"
PERCENT10 = "10percent"
PERCENT15 = "15percent"
PERCENT20 = "20percent"
PERCENT25 = "25percent"
PERCENT30 = "30percent"
#ALPHA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#BETA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# to test
methods = [ WAGNER ]
datasets = [FACEBOOK]
#alg_imports = [NODESIM,FAIRWALK,FIXEDCW,MINWALK,MINWALKNOP]
alg_imports = [NODE2VEC,PPR,ULOCALPPR,NLOCALPPR,PLOCALPPR]
minority_percentages = [PERCENT05]

# Constants for I/O
INPUT_DIR = "../input"
OUTPUT_DIR = "../1218"


def add_edges(nx_g, directed, edges):
    """
    Adds edges to given graph.
    """
    nx_g.add_edges_from(edges)
    if not directed:
        nx_g.add_edges_from([(v, u) for (u, v) in edges])


def remove_edges(nx_g, directed, edges):
    """
    Removes edges from given graph.
    """
    nx_g.remove_edges_from(edges)
    if not directed:
        nx_g.remove_edges_from([(v, u) for (u, v) in edges])


def get_minority_nodes(file_path):
    """
    Returns a list of minority nodes.
    """
    minorities = []
    with open(file_path, "r") as f:
        for line in f:
            node = int(line.strip())
            minorities.append(node)
    return minorities


def get_graph(file_path, directed):
    """
    Returns directed and undirected graphs as DiGraph.
    """
    init_g = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    if not directed:
        init_g.add_edges_from([(v, u) for (u, v) in init_g.edges()])
    return init_g


def evolve_network(nx_g, directed, minorities, recorder, method):
    """
    Iteratively evolve the network by adding and removing edges.
    """
    # initialize algorithm
    nx_g = alg.initialize(nx_g,directed=directed, protected=minorities)

    if VERBOSE:
        print("Iteration")
        print(f"0: {nx_g}")

    for i in range(1, ITERATIONS+1):
        if i==1:
            beginning = nx.pagerank(nx_g)
            sorted_nodes = sorted(beginning, key=beginning.get, reverse=True)
            top_10_percent = len(nx_g.nodes()) // 10
            top_nodes = sorted_nodes[:top_10_percent]
            VIS_M_Begin = len([node for node in top_nodes if node in minorities])/len(top_nodes)
        to_predict = method.nodes_to_predict(nx_g)
        predictions = alg.predict(nx_g, directed=directed, nodes=to_predict)

        # with open(predfile, 'w') as file:
        #     for tup in predictions:
        #         file.write(', '.join(map(str, tup)) + '\n')

        add_edges(nx_g, directed, predictions)

        # with open(graphfile, 'w') as f:
        #     for edge in nx_g.edges():
        #         f.write(f"{edge[0]} {edge[1]}\n")

        removals = method.edges_to_remove(nx_g)
        remove_edges(nx_g, directed, removals)

        # compute metrics
        recorder.record_metrics(nx_g.copy())

        initial = {}
        with open(f"../1218/{percent}/final_PR_node2vec_{basename}.txt", 'r') as file:
            for line in file:
                nodeid, nodevalue = line.strip().split(':')
                initial[int(nodeid)] = float(nodevalue)
        if i==ITERATIONS:
            name1 = f"../1218/{percent}/final_PR_{alg.NAME}_{basename}.txt"
            final = nx.pagerank(nx_g)
            with open(name1,"w") as f:
                for node,score in final.items():
                    f.write(f"{node}: {score}\n")
            sorted_nodes = sorted(final, key=final.get, reverse=True)
            top_10_percent = len(nx_g.nodes()) // 10
            top_nodes = sorted_nodes[:top_10_percent]
            VIS_M_End = len([node for node in top_nodes if node in minorities])/len(top_nodes)
            
            l1_norm = sum(abs(initial[node] - final[node]) for node in initial)
            l2_norm = math.sqrt(sum((initial[node] - final[node])**2 for node in initial))
            with open(f"../1218/{percent}/record.txt","a") as f1:
                f1.write(f"{alg.NAME}")
                f1.write(f"  {basename}")
                f1.write("\n")
                f1.write("Begin VIS_M: "+str(VIS_M_Begin)+"\n")
                f1.write("After VIS_M: "+str(VIS_M_End)+"\n")
                f1.write("VIS_M increased: "+str(VIS_M_End-VIS_M_Begin)+"\n")
                f1.write("l1_norm: "+str(l1_norm)+"\n")
                f1.write("l2_norm: "+str(l2_norm)+"\n\n")
        if VERBOSE and i % 2 == 0:
            print(f"{i}: {nx_g}")
    
    return nx_g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Evolution: Multiple Methods, Datasets, Algorithms")
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations")
    parser.add_argument("--continue", dest="reset", action="store_false", help="Continue evolution from last iteration")
    parser.add_argument("--silent", dest="verbose", action="store_false", help="Silence output")
    args = parser.parse_args()

    RESET_EVOLUTION = args.reset
    ITERATIONS = args.iterations
    VERBOSE = args.verbose

    for percent in minority_percentages:
        # output directory for this minority percentage
        OUTPUT_DIR = os.path.join(OUTPUT_DIR, percent)

        # create output directory if it doesn't exist
        # if not os.path.exists(OUTPUT_DIR):
        #     os.makedirs(OUTPUT_DIR)
    # for each method to test
        for method_import in methods:
            exec(method_import) # import the method

            # for each dataset to test
            for basename, directed in datasets:

                # for each algorithm to test
                for alg_import in alg_imports:
                    start = time.time()
                    exec(alg_import) # import the algorithm
                    
                    if VERBOSE:
                        print(f"----------------------------------------------")
                        print(f"Method: {Method.NAME}")
                        print(f"Dataset: {basename}")
                        print(f"Algorithm: {alg.NAME}")
                        #print(f"Algorithm: TestWalk_Alpha_{a}_Beta_{b}")
                        print(f"Minority Percentage: {percent}")

                    # file paths
                    ouput_prefix = f"{basename}.{Method.NAME}.{alg.NAME}"
                    minorities_path = os.path.join(INPUT_DIR, basename + ".minorities")
                    edgelist_path = os.path.join(INPUT_DIR, basename + ".txt")
                    evolved_edgelist_path = os.path.join(OUTPUT_DIR, ouput_prefix + ".txt")

                    minorities = get_minority_nodes(minorities_path)

                            # initialize recorder
                    recorder = Recorder(directed=directed,
                                        protected=minorities,
                                        output_dir=OUTPUT_DIR,
                                        output_prefix=ouput_prefix)
                        
                    if RESET_EVOLUTION:
                        init_g = get_graph(edgelist_path, directed)
                        recorder.clear_files()
                        recorder.record_metrics(init_g)
                    else:
                        try:
                            init_g = get_graph(evolved_edgelist_path, directed)
                        except FileNotFoundError:
                            print(f"Error: {evolved_edgelist_path} not found. Skipping dataset.")
                            continue # skip if not found
                        
                    # initialize method
                    method = Method(init_g, directed=directed, protected=minorities)

                    # evolve the network
                    final_g = evolve_network(init_g, directed, minorities, recorder, method)

                    # plot metrics
                    recorder.plot_metrics(show=False)

                    # write evolved graph to file
                    if directed:
                        nx.write_edgelist(final_g, evolved_edgelist_path, data=False)
                    else:
                        nx.write_edgelist(final_g.to_undirected(), evolved_edgelist_path, data=False)

                    end = time.time()
                    if VERBOSE:
                        print(f"Time elapsed: {end - start}")
