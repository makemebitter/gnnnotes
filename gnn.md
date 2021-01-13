### Intro

Data is graph

- graph can be irregular -> variable size of unordered nodes, variable number of edges -> convolution difficult
- i.i.d assumption no longer true, data points are explicitly related

### Central task
Figure out node embeddings and these embeddings should contain contributions from the node's neighbors, model both the nodes and it's dependencies with its neighbors

### Input

G(V, E)

A adjacency matrix

X node feature matrix

Xe edge feature matrix

Can be spatial-temporal if the graph/features vary over time.

### Output

- Node-level tasks
- Edge-level tasks
- Graph-level

### Learning

- Semi-supervised: nodes partially labeled
- Supervised: prediction on entire graph
- Un-supervised: graph embedding extraction

### Pre-GNN Graph kernel methods
Like SVM kernel tricks

### Recurrent GNN

Message passing (neighborhood aggregation) to consider neighbor nodes' contributions

```
Foward:
for each step do
    node embedding at current step = sum of f(node feature, edge feature, neighbor embedding at last step, neighbor feature) for each neighbor

H_t = F(H_t-1, X) -> H = F(H, X), when converged => H is the fixed point of F

f approximated with fully connected NN

guaranteed to converge if f is a contraction mapping:

d(f(x), f(y)) <= kd(x, y)
Backward:
Use another function g(embedding) to generate prediction target (labels) and calculate loss, backprop to train g and f. How? Naitvely BPTT, or Almeida–Pineda algorithm
```

MLP needs a correction term to satisfy this

find an equilibrium between all node embeddings

**Remark**: In general, the transition and the output functions and their parameters may depend on the node. In fact, it is plausible that different mechanisms (implementations) are used to represent different kinds of objects.

**Question**: how does `f` handle nodes with different number of neighbors? For positional graph where order of neighbors matter, initialize `f` with the maximum number of neighbors and padding. For non-positional graphs, simply aggregate (sum together) all neighbors contributions

**Learning**: a) initialize f, h_0. b) apply f to all nodes for multiple times to approach the equilibrium. c) calculate loss and gradients and apply to f to update the weights. d) repeat b-c)

**Examples**:

-   **GNN2009**, as described above
-   **GraphESN** (better efficiency by using the so-called echo states), 
-   **GGNN** use GRU so number of recurrent steps is controlled, no need for contraction mapping, even more in-efficient as it adopts BPTT for training 
-   **SSE** (stochastic and async by selecting subsets of nodes for forward/backward prop simultaneously, no theoretical guarantee to converge to fixed points)

**Problems**: recurrent functions multiple time -> intermediate states in memory if using BPTT, as it requires to store the states of every instance of the units. When the graphs and are large, the memory required may be considerable

What if the graph are disconnected graphs?

### Convolutional GNN

More efficient and more popular

#### Quick re-cap on convolutions (as in math)
```
- On multiplication of polynomial series
c, d two vectors in N-d functional space formed by x^0, x^1, x^2, ..., x^N, non-orthogonal
(c * d)_k = Sum c_i d_j, for all i + j = k
c = polynomial series (c_0 + c_1x + ... + c_Nx^N) 
d = (d_0 + d_1x + ... + d_Nx^N) 
if we do c x d
at the x^k direction the coefficient is
		c_0d_k + c_1d_(k - 1) + ..., which is the convolution of vector c and d
convolution = coefficients of the product of two polynomial series

- On two functions
(f * g)(t) = Integral f(t)g(t - x) for x from -inf to +inf

- two dimension
see below
```

![{\displaystyle \left({\begin{bmatrix}a&b&c\\d&e&f\\g&h&i\end{bmatrix}}*{\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}}\right)[2,2]=(i\cdot 1)+(h\cdot 2)+(g\cdot 3)+(f\cdot 4)+(e\cdot 5)+(d\cdot 6)+(c\cdot 7)+(b\cdot 8)+(a\cdot 9).}](https://wikimedia.org/api/rest_v1/media/math/render/svg/570600fdeed436d98626278f22bf034ff5ab5162)

note this is not the convolution used in CV world, CV convolution are actually cross-correlation

```
f conv g = (f complex conjugate) cross-correlation g, when f, g are functions
F conv G = (f complex conjugate) cross-correlation g, when F, G are matrices
```

**Check Mathematica note for definition of graph conv**

**How do we run the convolution?** Option 1 (Spatial-based): use the definition and directly calculate the results for each node. Option 2 (Spectral-based): Run Fourier transform, calculate the matrix multiplication, and transform back


#### Spectral-based

Stems from graph signal processing. Treat node features as signal. 

Represent graph as a Laplacian matrix L. Signal processing, Fourier transformation, convolution, etc. symmetric positive semidefinite => L = ULambdaU^T 

Nodes features are considered as signals and represented as matrix stacked up by vectors. 

```
x = N-d vector, N being number of nodes. Assme scalar feature for each node for now.
Question: apply a learnable filter on the signal. Calculate x * g ? Notice this filter is a global filter that filters on the entire data.
How to do it? We can use Fourier transform 

Fourier transform of x := F(x) = U^Tx
x * g = F^-1 F(x * g)
Convolution: 
x * g = U(U^T x point-wise product U^T g)
Convolution theorem: F(c star d) = F(c) .* F(d), .x: point-wise product, convolve and transform or transform then multiply, useful because Fourier transform can be fast, but not convolution. LHS: NlogN + N^2. RHS: 2NlogN + N
```

Spectral-based Conv: convolution is equivalent to multiplication in Fourier space, cross-correlation is equivalent to multiplication of complex conjugate of one matrix with the other

Problems: the graph must be static otherwise the eigenbasis will change and so does Fourier transformation => no simple way for transfer learning as the GNN is graph-dependent. 

Examples: Spectral CNN, ChebNet (approximate the filter by Chebyshev polynomials, confine the spectrum of the signal to be in [-1, 1], and filter calculation complexity is independent of the graph size), GCN(further approximation to ChebNet, both spectral and spatial)

#### Spatial-based

Generalization of image CNNs. Kind of similar to recurrent GNNs.

Recurrent GNN: h_0 -> f() -> h_1-> f() -> ...

Spatial-based Conv GNN: h_0 -> f_1() -> h_1 -> f_2() -> ...

Research focus seems to be on how to correctly weight contributions

```
H_n = f_n(
		f_(n-1)(
			f_(n-2)(X, H_(n-1), A)
		)
)
Propogation rule
Simplest: H_n = f_n(H_(n-1), A) = σ(A W_n H_(n-1)), σ is activation function, W_n 
Propogation rule.
```


Examples

**NN4G**, Node embedding = f(Wx + summation of (filtered neighbor embeddings)), f learnable, filter learnable

**DCNN**, diffusion model, node embedding = activation(W .x (PX))

P diffusion matrix = transition probability = D^(-1)A, X = node features stacked together, this model means the message would not likely to flow to distant nodes

**PGC-DGCNN **diffusion model, increases the contributions of distant neighbors based on shortest paths

**MPNN** is a general framework of spatial-based convGNNs. General form of expression:

![screenshot](/Users/YHZhang/Library/Group Containers/Q79WDW8YH9.com.evernote.Evernote/Evernote/quick-note/50159951-personal-www.evernote.com/quick-note-i8Mxbq/attachment--31mHjC/screenshot.png)

```
h_vk, embedding of node v at layer k
U_k, parameterized function at layer k
M_k, parameterized function at layer k
N(v), neighbors of node v
u, neighbor of node v
xe_vu, edge feature of v-u
```



**GIN** adds yet another learnable weight to distinguish between the node's own embedding and its neighbor's embeddings, note it still assumes identical weights (these weights are for the weighted sum of neighbor embeddings) for all neighbors

**GraphSage** addresses scalability issues (many neighbors) by introducing sampling over the neighbors

**GAT** attention model to learn about the contribution of neighbors, major difference between this and other non-parameteric weight-based methods like GCN. It also further uses multi-head attention. There are still many graph attention mechanism work: **GAAN**(weighted multi-head), **GeniePath**(LSTM-like)

**On manifolds**: GCNN, ACNN, MoNet

**PATCHY-SAN** rank neighbors by graph labelings and select top q, graph becomes grid, preprocessing heavy

**LGCN** rank neighbors by node feature

##### Problems: 

-   High memory footprint: usually in-memory and intermediate states are large. ~million nodes. **GraphSage** has sampling of neighbors to avoid this by hierarchical aggregations, trades time for space. **Fast-GCN** samples nodes instead. **Cluster-GCN** samples sub-graph. **StoGCN** reduces receptive field of the graph convolution to an arbitrarily small scale using historical node representations as a control variate. Still has high memory consumption as historical embeddings needed and this method is very slow in terms of time complexity. 

##### Spatial ConvGNNs are preferred over Spectral ones

-   More efficient. No need to calculate eigenvectors or calculate global information, can run mini-batch learning
-   More generalizable. Spectral graphs are heavily bound by the assumption of static graphs as per the request of the Fourier transformation
-   Can work on directed graphs and multi-source graph inputs like edges, singed graphs

### Down-sampling layers
Problem: there are many nodes and their embeddings, very high dimention. Not practical to use consider all of them. Main focus of research is to  improve the effectiveness and computational
complexity of pooling 

-   Pooling to reduce params, mean/max/sum, problem is fixed size embedding is not efficient for smaller graphs

-   Readout to generate graph-level representation

**Examples:** Set2Set, dynamic graph embedding size. ChebNet, DGCNN, sortpooling by sort the nodes first by features then truncate/pad. DiffPool, SAGPool, consider structral information of graph when conducting pooling.

### Miscs
#### VC dimention
For GNN2009, p: number of parameters, n: number of graph nodes O(p^4n^2)
#### Graph isomorphism (check if two graphs are topologically identical)
Some research shows that common GNNs are not capable of distinguishing graph structures

### GNN Autoencoders (GAEs)

#### Graph embedding
Learn and represent graph's topological information such as PPMI matrix and adjacency matrix. PPMI (Positive Pointwise Mutual Information): log(P(x,y) / P(x)P(y), first-order co-occurrance: nearby each other. second-order: have similar neighbors

NOTE: Not identical to computer vision auto encoders as those decode directly to images. But here we are modeling only the PPMI matrix.

**Focus**: Link prediction problem. 
Examples: 

-   **DNGR**, encode and decode PPMI matrix., 
-   **SDNE**, encode first order and second order information jointly. 
-   **GAE2016**, use GCN to encode both structural and feature information. 
-   **VGAE**, variational autoencoder. 
-   **ARVGA** GAN. 
-   **DRNE**, converts graph into random walk sequences, assumes a nodes' embeddings should be aggregation of neighbor embeddings. Uses LSTM to aggregate neighbors, convert a sequence of node's neighbor to a single embedding (sequence to word). **NetRA**.

**Problems:** In the sparse graphs, input is in-balanced where positive links are far less than negative ones. Scalability issues

#### Graph generation
Majority of applications: molecular graph generation problem especially in drug discoverty.
Examples: 

-   **SMILES**, nodes and edges step by step using a string representation of molecular formula, domain specific. 
-   **DeepGMG**: general framework, assumes given a graph embedding, total probability of the graph is the sum of all node permutations. DeepGMG generates graphs by making a sequence of decisions, namely whether to add a node, which node to add, whether to add an edge, and which node to connect to the new node. The decision process of generating nodes and edges is conditioned on the node states and the graph state of a growing graph updated by a RecGNN. 
-   **GraphRNN**, separate the decision process for nodes and edges using two RNNs.
-    **GraphVAE** generates graph all at once (adjacency matrix, node features and edge features). Hard to make sure the graph generated is valid.
-   **RGVAE** regularizes the output for validity. 
-   **MolGAN** adopts reinforcement learning for generating graph with desired properties.

**Problems**: sequential ones linearize graph to sequences. Loses structural information because of CYCLES. Global approaches generate graph all at once, scalability issue for giant graphs.

### Spatial-temporal GNN

Dynamic graphs.

#### RNN based

Capture spatial-temporal dependencies by filtering inputs and hidden states passed to a recurrent unit using graph convolutions, RNN upon CNN

**Examples: ** GCRN, DCRNN. Structural-RNN uses a node RNN an edge RNN to capture information separately for each node and edge, very expensive so cluster nodes and edges into semantic groups.

**Problems: **Time consuming, gradient explosion/vanishing.

#### CNN based

More efficient and stable than RNN based.

Use 1-D CNN, kind of like the 4-D CNNs used in videos.

**Examples: **CGCN, ST-GCN, these two assume adjacency matrix available. Graph WaveNet, GaAN, ASTGCN predict graph structure by learning a latent static graph structure. It uses node embeddings to predict dependencies.

**Problem for the dependency prediction ones: ** Need to calculate every single pair of nodes to get the dependencies, which is very slow.

### Applications

#### Datasets

-   Citation networks, pubmed, dblp ~4M nodes, 36M edges
-   Biochemical graphs. Usually up to several hundreds of nodes and edges, thousands of graphs. Largest is PPI, 50K nodes, 800K edges.
-   Social networks. Reddit 200K nodes, 11M edges. ?Facebook ?twitter

#### Prediction tasks

-   Node classification. Normal train/valid/test split. Average accuracy/F1 score.
-   Graph Classification. 10-fold cross validation. Some controversy
#### Implementation
PyTorch Geometric: many GNNs implementated
Deep Graph Library: PyTorch and MXnet, fast implementation

Tensorflow and PyTorch are the most used

#### Domain-specific usage
- Computer vision:

  -   Recognizing semantic relationships between objects, scene graph generation, or reverse the process to generate fake images. 
  -   Synthesize image given textual descriptions. 
  -   Points cloud segmentation in LiDAR data.
  -   Human action classification in video, detect joints and form a graph
-  NLP

    -  Text data, aside from the temporal relationship, can also have graphical relationship. e.g. syntactic dependency tree. Syntactic GCN

    -   Abstract Meaning Representation to sentence. Generate a sentence given a semantic graph. Graph-LSTm, GGNN. Machine translation
- Traffic
- Recommender systems
    - Items-items, users-users, items-users relations.
- Chemistry and MSE

### Potential system issues
- Many ways to build the graph, unified API?
- Data preprocessing very important, especially for the spectral models
- Mostly in-memory processing, scalability in graph with million nodes?
- Efficiency is one of the top problems, many models are still slow. Sampling? Clustering? Trade-offs?
- For ConvGNN Deeper nets != better accuracy. As data is connected, theoretically infinite GNN layer will consider the whole graph and learns a global embedding for all nodes
- How to run inference? Dynamic graph? When one more node is added, do you need to re-calculate all the rest nodes' embeddings?
- What about multi-modal analysis? Images, texts mixed?
- Are there redundancies, is it possible to optimize them without affecting the accuracy?







