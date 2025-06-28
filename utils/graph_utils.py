import json
import networkx as nx
import plotly.graph_objects as go


def load_graph_from_file(file_path: str, throw_error=False) -> nx.DiGraph:
    with open(file_path, "r") as json_file:
        graph_dict = json.load(json_file)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for node in graph_dict["nodes"]:
        G.add_node(node["id"], label=node["label"])

    # Add edges
    for link in graph_dict["links"]:
        G.add_edge(link["source"], link["target"])

    
    root_id = [node for node, degree in G.in_degree() if degree == 0]

    if throw_error and len(root_id) != 1:
        raise ValueError(f"There should be exactly one root node in the hierarchy. Found {len(root_id)}")
    else:
        if any([isinstance(s, str) for s in root_id]):
            root_id = root_id[0]  # If root_id is a list with one string, take that string
        else:
            root_id = max(root_id)

    return G, root_id

# below code from hulikalruthu

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

    print(f"links: {data['links']}")

    print(f"added {len(G.nodes())} nodes and {len(G.edges())} edges to the graph")
    # Get labels for nodes
    labels = nx.get_node_attributes(G, 'label')
    print(f" got {len(labels)} labels for the nodes")

    # Detect circular references
    cycles = list(nx.simple_cycles(G))
    if cycles:
        print("Circular references detected!")
        for cycle in cycles:
            print(f"Cycle: {cycle}")
    else:
        print("No circular references detected.")

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

    # fig.show()
    # save as png
    fig.write_image(f"{datasetname}_hierarchy_graph_plotly.png")
    # save as html as well
    fig.write_html(f"{datasetname}_hierarchy_graph_plotly.html")