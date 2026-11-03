# -*- coding: utf-8 -*-
"""
SOCIAL NETWORK ANALYSIS (SNA): METRICS, CENTRAL ACTORS AND COMMUNITIES
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the Social Network Analysis course scripts (1_SNA_conhecendo_codigos,
2 and 3_SNA_exemplo_comunidades_base_Karate).

GOAL
- From an EDGE LIST (who relates to whom: e-mails, transfers, co-authorships,
  referrals, joint purchases...), compute:
  * network metrics (density, diameter, average path length, components);
  * centrality of each actor (degree, betweenness, closeness, eigenvector,
    PageRank, clustering coefficient, eccentricity);
  * communities with several methods, choosing the one with the highest
    MODULARITY.

INPUT
- File with one row per edge: source and target columns (and, optionally,
  weight). You can specify whether the network is directed.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy networkx scipy matplotlib openpyxl
"""

#%% CONFIGURATION (the only place to edit)

FILE = "karate.csv"
SHEET = None
SEPARATOR = None
SOURCE_COLUMN = "Source"
TARGET_COLUMN = "Target"
WEIGHT_COLUMN = None            # e.g. "amount" or "n_messages" (None = unweighted)
DIRECTED = "auto"               # True, False or "auto" (reads the "Type" column, if any)
REMOVE_SELF_LOOPS = True        # ignores edges from a node to itself
COMMUNITY_METHODS = ["louvain", "greedy_modularity", "label_propagation",
                     "girvan_newman"]
TOP_N = 10
SEED = 42

OUTPUT_FOLDER = "results"
SHOW_PLOTS = True
MAX_PLOT_NODES = 500            # above this, draws only the top nodes by degree


#%% Packages

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community as nxcom

warnings.filterwarnings("ignore")


#%% Utility functions (shared by all tools; no need to edit)

SCRIPT_DIR = (Path(__file__).resolve().parent
              if "__file__" in globals() else Path.cwd())
REPORT = []


def resolve_path(relative):
    """Turn a path relative to the script folder into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def read_data(file, sheet=None, separator=None):
    """
    Read CSV, Excel, Parquet, Pickle or JSON based on the file extension.

    For CSV, the separator is detected and the most common encodings are tried
    (utf-8, utf-8 with BOM and latin-1), avoiding errors with accented text.
    """
    path = resolve_path(file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    df = None
    if ext in (".csv", ".txt"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, sep=separator, engine="python", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    elif ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # Remove spaces and invisible marks (BOM) from column names
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def prepare_output(folder):
    """Create (if needed) and return the results folder."""
    output = resolve_path(folder)
    output.mkdir(parents=True, exist_ok=True)
    return output


def log(text=""):
    """Print to the console and keep the line for report.md."""
    print(text)
    REPORT.append(str(text))


def save_figure(fig, name):
    """Save the chart as PNG in the output folder and show it (if configured)."""
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=120, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_report(title):
    """Write all logged lines to report.md."""
    text = f"# {title}\n\n" + "\n".join(REPORT) + "\n"
    (OUTPUT / "report.md").write_text(text, encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT / 'report.md'}")


def md_table(df, decimals=3):
    """Convert a DataFrame into a Markdown table (without depending on tabulate)."""
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return ""
            if x != 0 and abs(x) < 10 ** -decimals:
                return f"{x:.2e}"
            return f"{round(float(x), decimals):g}" if abs(x) < 1e6 else f"{x:,.0f}"
        return x
    d = df.copy()
    d = d.reset_index()
    if d.columns[0] == "index":
        d = d.rename(columns={"index": ""})
    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |"
              for r in d.astype(object).map(fmt).values]
    return "\n".join(lines)


OUTPUT = prepare_output(OUTPUT_FOLDER)


#%% 1. Build the network

edges = read_data(FILE, SHEET, SEPARATOR)
for col in (SOURCE_COLUMN, TARGET_COLUMN):
    if col not in edges.columns:
        raise KeyError(f"Column '{col}' not found: {list(edges.columns)}")
edges = edges.dropna(subset=[SOURCE_COLUMN, TARGET_COLUMN])
if DIRECTED == "auto":
    type_col = next((c for c in edges.columns if c.lower() == "type"), None)
    directed = bool(type_col and edges[type_col].astype(str).str.lower()
                    .str.startswith("dir").mean() > 0.5)
else:
    directed = bool(DIRECTED)

G = nx.DiGraph() if directed else nx.Graph()
for _, e in edges.iterrows():
    s, t = e[SOURCE_COLUMN], e[TARGET_COLUMN]
    if REMOVE_SELF_LOOPS and s == t:
        continue
    weight = float(e[WEIGHT_COLUMN]) if WEIGHT_COLUMN else 1.0
    if G.has_edge(s, t):
        G[s][t]["weight"] += weight   # repeated edges add up their weight
    else:
        G.add_edge(s, t, weight=weight)
weight_arg = "weight" if WEIGHT_COLUMN else None
undirected = G.to_undirected() if directed else G

log("## 1. Network")
log(f"- File: `{FILE}` | {'directed' if directed else 'undirected'}"
    f"{' | weighted' if WEIGHT_COLUMN else ''}")
log(f"- Nodes (actors): {G.number_of_nodes()} | edges: {G.number_of_edges()}")


#%% 2. Network metrics
# Density = existing edges / possible edges. Diameter = largest distance
# between two actors. Transitivity = chance that "friends of friends" are friends.

components = sorted(nx.connected_components(undirected), key=len, reverse=True)
largest = undirected.subgraph(components[0]).copy()
network_metrics = {
    "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
    "density": nx.density(G),
    "mean_degree": np.mean([g for _, g in undirected.degree()]),
    "components": len(components),
    "nodes_in_largest_component": len(components[0]),
    "transitivity": nx.transitivity(undirected),
    "mean_clustering": nx.average_clustering(undirected),
}
if len(largest) <= 5000:
    network_metrics["largest_component_diameter"] = nx.diameter(largest)
    network_metrics["largest_component_mean_path"] = nx.average_shortest_path_length(largest)
if directed:
    network_metrics["reciprocity"] = nx.reciprocity(G)
network_tab = pd.Series(network_metrics, name="value").to_frame()
log("\n## 2. Network metrics")
log(md_table(network_tab, 4))


#%% 3. Actor centrality
# Degree: number of connections (popularity). Betweenness: how many shortest
# paths go through the actor (bridge/flow control).
# Closeness: how close the actor is to everyone (fast reach).
# Eigenvector/PageRank: connected to actors who are also important (influence).

k_sample = None if G.number_of_nodes() <= 2000 else 500
centrality = pd.DataFrame({
    "degree": dict(undirected.degree(weight=weight_arg)),
    "normalized_degree": nx.degree_centrality(undirected),
    "betweenness": nx.betweenness_centrality(G, weight=weight_arg, k=k_sample,
                                             seed=SEED),
    "closeness": nx.closeness_centrality(G),
    "pagerank": nx.pagerank(G, weight=weight_arg),
    "clustering": nx.clustering(undirected),
})
try:
    centrality["eigenvector"] = pd.Series(nx.eigenvector_centrality_numpy(undirected,
                                                                          weight=weight_arg))
except Exception:
    centrality["eigenvector"] = np.nan
if directed:
    centrality["in_degree"] = pd.Series(dict(G.in_degree(weight=weight_arg)))
    centrality["out_degree"] = pd.Series(dict(G.out_degree(weight=weight_arg)))
eccentricity = nx.eccentricity(largest) if len(largest) <= 5000 else {}
centrality["eccentricity"] = pd.Series(eccentricity)
centrality.index.name = "actor"

log(f"\n## 3. Most central actors (top {TOP_N})")
for measure in ["degree", "betweenness", "closeness", "pagerank"]:
    top = centrality[measure].sort_values(ascending=False).head(TOP_N)
    log(f"- **{measure}**: " + ", ".join(f"{i} ({v:.3g})" for i, v in top.items()))


#%% 4. Communities
# Modularity (-0.5 to 1): measures whether there are more edges WITHIN the
# communities than expected by chance. Above ~0.3 indicates a clear community
# structure.

methods = {
    "louvain": lambda: nxcom.louvain_communities(undirected, weight=weight_arg, seed=SEED),
    "greedy_modularity": lambda: nxcom.greedy_modularity_communities(undirected, weight=weight_arg),
    "label_propagation": lambda: list(nxcom.label_propagation_communities(undirected)),
    "girvan_newman": lambda: max(
        (c for _, c in zip(range(min(30, undirected.number_of_nodes())),
                           nxcom.girvan_newman(undirected))),
        key=lambda c: nxcom.modularity(undirected, c)),
}
community_results = {}
for name in COMMUNITY_METHODS:
    if name == "girvan_newman" and undirected.number_of_edges() > 2000:
        log("- girvan_newman skipped (large network: cost too high).")
        continue
    try:
        parts = [set(c) for c in methods[name]()]
        community_results[name] = (nxcom.modularity(undirected, parts, weight=weight_arg), parts)
    except Exception as error:
        log(f"- {name} failed: {error}")

community_tab = pd.DataFrame({n: {"modularity": m, "n_communities": len(p),
                                  "largest_community": max(len(c) for c in p),
                                  "smallest_community": min(len(c) for c in p)}
                              for n, (m, p) in community_results.items()}).T \
    .sort_values("modularity", ascending=False)
best = community_tab.index[0]
partition = community_results[best][1]
community_map = {node: i + 1 for i, c in enumerate(sorted(partition, key=len, reverse=True))
                 for node in c}
centrality["community"] = pd.Series(community_map)

log("\n## 4. Communities")
log(md_table(community_tab, 4))
log(f"- Method chosen (highest modularity): **{best}**")
for i, c in enumerate(sorted(partition, key=len, reverse=True), start=1):
    leader = centrality.loc[list(c), "degree"].idxmax()
    log(f"- Community {i}: {len(c)} actors; most connected = {leader}")


#%% 5. Visualization

H = G if G.number_of_nodes() <= MAX_PLOT_NODES else G.subgraph(
    centrality["degree"].sort_values(ascending=False).head(MAX_PLOT_NODES).index)
position = nx.spring_layout(H, seed=SEED, weight=weight_arg)
colors = [centrality.loc[n, "community"] for n in H.nodes()]
sizes = 80 + 1500 * centrality.loc[list(H.nodes()), "normalized_degree"].values
fig, ax = plt.subplots(figsize=(11, 8))
nx.draw_networkx_edges(H, position, alpha=0.25, ax=ax,
                       arrows=directed, arrowsize=8)
nx.draw_networkx_nodes(H, position, node_color=colors, cmap="tab10",
                       node_size=sizes, ax=ax)
if H.number_of_nodes() <= 80:
    nx.draw_networkx_labels(H, position, font_size=8, ax=ax)
ax.set_title(f"Network: color = community ({best}), size = degree")
ax.axis("off")
save_figure(fig, "01_network_communities")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(centrality["degree"], bins=min(30, centrality["degree"].nunique()),
        color="steelblue")
ax.set(title="Degree distribution", xlabel="Degree (number of connections)",
       ylabel="Number of actors")
save_figure(fig, "02_degree_distribution")


#%% 6. Save

centrality.sort_values("degree", ascending=False).to_csv(
    OUTPUT / "actor_metrics.csv", encoding="utf-8-sig")
network_tab.to_csv(OUTPUT / "network_metrics.csv", encoding="utf-8-sig")
community_tab.to_csv(OUTPUT / "community_comparison.csv", encoding="utf-8-sig")
nx.write_gexf(G, OUTPUT / "network.gexf")   # opens in Gephi

log("\n## 5. How to interpret")
log("- High betweenness and low degree: a 'bridge' between groups (a risk if "
    "it leaves, an opportunity to spread information).")
log("- High PageRank/eigenvector: influence from being connected to "
    "influential actors.")
log("- Communities: groups more connected among themselves than with the rest "
    "(informal teams, customer segments, collusive fraud).")

save_report("Report: Social Network Analysis")
