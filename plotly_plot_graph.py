# Code to plot a sunburst graph from hierarchy.json files.
# Requires pip or conda install libraries networkx, plotly and kaleido

import json
import networkx as nx
import plotly.graph_objects as go
import kaleido


def plot_hierarchy(json_file, datasetname="dataset"):
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create a directed graph
    G = nx.DiGraph()
    print(f"Loaded graph and data")

    # Add nodes
    for node in data["nodes"]:
        G.add_node(node["id"], label=node["label"])

    # Add edges
    for link in data["links"]:
        G.add_edge(link["source"], link["target"])

    print(f"added {len(G.nodes())} nodes and {len(G.edges())} edges to the graph")
    # Get labels for nodes
    labels = nx.get_node_attributes(G, 'label')
    print(f" got {len(labels)} labels for the nodes")

    # Generate a radial tree using Plotly's Sunburst chart
    ids = []
    labels_list = []
    parents = []

    for node in G.nodes():
        ids.append(node)
        labels_list.append(labels[node])
        # Find the parent of the node
        parent = next(iter(G.predecessors(node)), "")  # Root node has no parent
        parents.append(parent)

    # Create the Sunburst chart
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels_list,
        parents=parents,
        branchvalues="total"
    ))

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0)
    )

    fig.show()
    # save as png
    fig.write_image(f"{datasetname}_hierarchy_graph_plotly.png")

# Example usage
# plot_hierarchy("./madverse/madverse_output_hierarchy.json", datasetname="madverse")
plot_hierarchy("./grocerydataset/hierarchy.json", datasetname="grocery")
