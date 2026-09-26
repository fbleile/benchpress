#!/usr/bin/env python3
"""Plot the bundled Sachs ground-truth DAG."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth", type=Path,
        default=Path("resources/adjmat/myadjmats/sachs.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/sachs_truth_graph.png"),
    )
    args = parser.parse_args()

    table = pd.read_csv(args.truth)
    names = list(table.columns)
    adjacency = table.to_numpy(dtype=bool)
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_edges_from(
        (names[parent], names[child])
        for parent, child in zip(*adjacency.nonzero())
    )

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Sachs truth graph is not acyclic")

    roots = {node for node in graph if graph.in_degree(node) == 0}
    position = nx.spring_layout(graph, seed=17, k=1.4, iterations=300)
    node_colors = ["#F8766D" if node in roots else "#00BFC4" for node in graph]

    fig, axis = plt.subplots(figsize=(11, 8.5))
    nx.draw_networkx_nodes(
        graph, position, ax=axis, node_color=node_colors,
        node_size=1500, edgecolors="white", linewidths=1.5,
    )
    nx.draw_networkx_labels(
        graph, position, ax=axis, font_size=10,
        font_weight="bold", font_color="#202020",
    )
    nx.draw_networkx_edges(
        graph, position, ax=axis, arrows=True, arrowstyle="-|>",
        arrowsize=24, width=1.8, edge_color="#444444",
        connectionstyle="arc3,rad=0.08", min_source_margin=18,
        min_target_margin=18,
    )
    axis.set_title("Sachs ground-truth DAG (17 edges)", fontsize=15, pad=14)
    axis.text(
        0.01, 0.01,
        "Rows are parents and columns are children; red = source nodes",
        transform=axis.transAxes, fontsize=9, color="#444444",
    )
    axis.axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
