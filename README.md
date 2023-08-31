# Recommendation Algorithms Affect on Graph Evolution
## Useage

Multiple datasets, algorithms and methods can be evaluated using either the `Driver_Multi_Evolution.ipynb` notebook or the `driver_multi_evolution.py` script.

To run:

1. Install python3 and necessary packages (networkx, numpy, gensim, leidenalg, matplotlib)
    * Environment setup with conda:
    `conda create --name link-rec-evolution python=3.11 networkx numpy gensim leidenalg matplotlib`
    `conda activate link-rec-evolution`
    `conda install -c conda-forge leidenalg`
    `conda install -c conda-forge python-igraph`
2. cd into the src directory
3. run: `python3 driver_multi_evolution.py`

This will run a default of 30 iterations using the methods, algorithms and datasets specified in `driver_multi_evolution.py`.
The output will be created in the `output` folder with the naming convention: `[dataset].[method].[algorithm].[datatype]`.
For instance the visibility data for the `facebook` dataset using a method called `method1` and `node2vec` algorithm will have the file name `facebook.method1.node2vec.vis`.

The number of iterations can be changed using the --iterations flag:

`python3 driver_multi_evolution.py --iterations 10`

And a previous round of evolutions can be continued by using the --continue flag:

`python3 driver_multi_evolution.py --iterations 10 --continue`


## Datasets

The `input` folder contains several large social networks found at the [Stanford Large Network Dataset Collection](http://snap.stanford.edu/data/index.html).
The networks are stored as files of edges, where each row is and edge (node pairs separated by a space). For directed graphs the source node is the first node of the pair.

### Undirected Networks
* [Social circles: Facebook](http://snap.stanford.edu/data/egonets-Facebook.html) - 4,039 Nodes, 88,234 Edges
* [Deezer Europe Social Network](http://snap.stanford.edu/data/feather-deezer-social.html) - 28,281 Nodes, 92,752 Edges
* [LastFM Asia Social Network](http://snap.stanford.edu/data/feather-lastfm-social.html) - 7,624 Nodes, 27,806 Edges

### Directed Networks
* [email-Eu-core network](http://snap.stanford.edu/data/email-Eu-core.html) - 1,005 Nodes, 25,571 Edges
* [Wikipedia vote network](http://snap.stanford.edu/data/wiki-Vote.html) - 7,115 Nodes, 103,689 Edges
* [US Congress Twitter](http://snap.stanford.edu/data/congress-twitter.html) - 475 Nodes, 13,289 Edges


## Output Data

Five types of data are output.

1. **Visibility**
    * .vis
    * This is a measure of the fraction of minority nodes in the top 10% of Pagerank for directed graphs and Eigenvector Centrality for undirected graphs.
2. **Gini of the In Degree**
    * .gin
    * This is the gini of the in degree (number of incoming edges) for the graphs nodes.
3. **Averaged Clustering**
    * .clu
    * The clustering coefficient for the graph.
4. **Summery Plot**
    * .png
    * A plot that summarizes the three measurements.
5. **Edgelist of the Final Graph**
    * .txt
    * This is the edgelist of the graph after evolution is completed. Used to continue graph evolution.


## Generating Minorities

Minorities nodes are stored in the `.minorities` file associated with each dataset in the `input` folder.

To generate a `.minorities` file for a new dataset first add an edgelist file for the new dataset to the `input` folder.
Then execute the `Minority_Detection.ipynb` notebook in the `src/utils` folder with the name of the new edgelist added to the 
`GRAPHS` list. This will find the minorities nodes for the graph.

**Minority Definition for this work**

Minority nodes are found using structural community detection algorithms (louvain for undirected and leidenalg for directed).
Nodes are sorted by the size of the community they belong to, then the lower fraction of these nodes are taken to be the minorities.
This faction is set using the `MINORITY_RATIO` in `Minority_Detection.ipynb`. Nodes belonging to communities at or below a given size are not considered
in this process. This size is specified by the `COMM_THRESHOLD` in `Minority_Detection.ipynb`.

**NOTE** This work assumes that existing graph structure is a good indication of minority communities and that these communities can be accurately detected by 
community detection algorithms.