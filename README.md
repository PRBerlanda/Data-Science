# 16 · Social Network Analysis (SNA)

A ready-to-use tool that analyzes a **network of relationships** from an edge list (who is connected to whom). It computes network-level metrics, ranks actors by several centrality measures, detects communities with four algorithms, keeps the one with the highest modularity, and exports a picture of the network plus a file for Gephi.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Social network analysis studies the **structure** of relationships rather than the attributes of individuals. Each actor is a node, and each relationship is an edge, optionally directed (A → B) and weighted (number of messages, amount transferred). From that structure, the tool answers three questions:

1. **What does the network look like as a whole?** How dense, connected and clustered it is.
2. **Who is central?** Popular, bridging, close to everyone, or connected to other important actors.
3. **Which groups exist?** Communities of actors more connected among themselves than with the rest.

## When to use it

- E-mail or chat networks inside a company (informal leaders, silos).
- Financial transfers (fraud rings, money flows), co-authorship, referrals, joint purchases.
- Supplier/customer networks, social media interactions, citation networks.

## How it works

1. **Build the network** from the source and target columns, either directed or undirected (`"auto"` reads a `Type` column if present). Self-loops can be removed, and repeated edges add up their weight.
2. **Network metrics**: nodes, edges, density, mean degree, connected components, transitivity, mean clustering, diameter and average shortest path of the largest component, and reciprocity for directed networks.
3. **Centrality of each actor**: degree, normalized degree, betweenness, closeness, PageRank, eigenvector, clustering coefficient, eccentricity, and in/out degree for directed networks.
4. **Communities** with Louvain, greedy modularity, label propagation and Girvan-Newman (skipped for large networks). The partition with the **highest modularity** is chosen, and the most connected actor of each community is reported.
5. **Visualization**: network colored by community with node size proportional to degree, plus the degree distribution.
6. **Export** of the metrics and a `.gexf` file to open in Gephi.

## Inputs

**Data:** an edge list, with one row per relationship and columns for source, target and, optionally, weight and type.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator (auto-detected). |
| `SOURCE_COLUMN`, `TARGET_COLUMN` | Columns with the two ends of each edge. |
| `WEIGHT_COLUMN` | Optional edge weight (`None` = unweighted). |
| `DIRECTED` | `True`, `False` or `"auto"`. |
| `REMOVE_SELF_LOOPS` | Ignore edges from a node to itself. |
| `COMMUNITY_METHODS` | Community algorithms to compare. |
| `TOP_N` | Number of top actors listed per centrality measure. |
| `SEED` | Reproducibility of layouts and stochastic algorithms. |
| `MAX_PLOT_NODES` | Above this size, only the top nodes by degree are drawn. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Network summary, metrics, most central actors, community comparison and interpretation. |
| `network_metrics.csv` | Network-level metrics. |
| `actor_metrics.csv` | All centrality measures and the community of each actor. |
| `community_comparison.csv` | Modularity, number and size of communities for each method. |
| `network.gexf` | The network, ready to open in Gephi. |
| `01_network_communities.png`, `02_degree_distribution.png` | Network picture and degree histogram. |

## How to read the results

**Network level**
- **Density**: share of possible edges that exist (0 to 1). Real networks are usually sparse.
- **Components**: separate pieces of the network. The largest component holds most actors in a connected network.
- **Diameter / average path**: the longest and the typical number of steps between two actors. Short paths mean information spreads quickly.
- **Transitivity / clustering**: the chance that two contacts of the same actor are also connected ("friends of friends").

**Actor level**
- **Degree**: number of connections (popularity, activity).
- **Betweenness**: how often the actor lies on the shortest path between others. These are bridges that control the flow between groups. **High betweenness with low degree** is a classic bridge: a risk if the actor leaves, and an opportunity to spread information.
- **Closeness**: how quickly the actor can reach everyone else.
- **Eigenvector / PageRank**: influence from being connected to other influential actors.

**Communities**
- **Modularity** (−0.5 to 1): above about 0.3 indicates a clear community structure.
- Communities often reveal informal teams, customer segments or collusive groups. The most connected actor of each community is a natural point of contact.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your edge list next to `social_network_analysis.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE`, `SOURCE_COLUMN` and `TARGET_COLUMN`).
3. Run the script in Spyder (F5) or with `python social_network_analysis.py`.
4. Open `results/report.md`, or `results/network.gexf` in Gephi.

## Example dataset

`karate.csv` is Zachary's karate club network, a classic in network science: friendships between members of a university karate club. It uses semicolon-separated `Source`, `Target` and `Type` columns, and all edges are undirected.

## Credits

Built from the Social Network Analysis course scripts (MBA in Data Science and Analytics, USP/Esalq).
